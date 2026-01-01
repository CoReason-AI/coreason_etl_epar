import polars as pl

from coreason_etl_epar.transform_silver import clean_epar_bronze


def test_clean_epar_invisible_chars() -> None:
    """
    Test removal of invisible characters (\u200b).
    """
    data = {
        "product_number": ["EMEA/H/C/001"],
        "medicine_name": ["Medicine\u200bA"],  # Invisible char
        "active_substance": ["Substance\u200bB"],
        "atc_code": ["A01BC01\u200b"],
        "authorisation_status": ["Authorised\u200b"],
        "category": ["Human"],
    }
    df = pl.DataFrame(data)

    cleaned = clean_epar_bronze(df)

    # Assertions
    # \u200b should be gone from string columns
    assert cleaned["medicine_name"][0] == "MedicineA"
    assert "active_substance_list" in cleaned.columns
    assert cleaned["active_substance"][0] == "SubstanceB"
    assert cleaned["atc_code"][0] == "A01BC01"
    assert cleaned["authorisation_status"][0] == "Authorised"


def test_clean_epar_normalization() -> None:
    """
    Test normalization of Substance, ATC, Status, and Base ID.
    """
    data = {
        "product_number": ["EMEA/H/C/001234"],
        "active_substance": ["Sub A + Sub B/Sub C"],  # Mixed delimiters
        "atc_code": ["A01BC01, B01AB01; INVALID"],  # Delimiters and Invalid
        "authorisation_status": ["Conditional Approval"],
        "medicine_name": ["Med"],
    }
    df = pl.DataFrame(data)

    cleaned = clean_epar_bronze(df)

    # 1. Base ID
    assert cleaned["base_procedure_id"][0] == "001234"

    # 2. Substance (List)
    subs = cleaned["active_substance_list"][0].to_list()
    assert sorted(subs) == ["Sub A", "Sub B", "Sub C"]

    # 3. ATC Code (List)
    atcs = cleaned["atc_code_list"][0].to_list()
    assert "A01BC01" in atcs
    assert "B01AB01" in atcs
    assert "INVALID" not in atcs  # Should be filtered out by strict regex

    # 4. Status
    assert cleaned["status_normalized"][0] == "CONDITIONAL_APPROVAL"


def test_clean_epar_status_normalization() -> None:
    """
    Test status normalization logic via clean_epar_bronze.
    """
    data = {
        "product_number": [f"P{i}" for i in range(7)],
        "medicine_name": ["M"] * 7,
        "authorisation_status": [
            "Authorised",
            "Refused",
            "Withdrawn",
            "Suspended",
            "Exceptional Circumstances",
            "Authorised under exceptional circumstances",
            "Conditional Marketing Authorisation",
        ],
    }
    df = pl.DataFrame(data)
    cleaned = clean_epar_bronze(df)
    results = cleaned["status_normalized"].to_list()

    expected = [
        "APPROVED",
        "REJECTED",
        "WITHDRAWN",
        "SUSPENDED",
        "EXCEPTIONAL_CIRCUMSTANCES",
        "EXCEPTIONAL_CIRCUMSTANCES",
        "CONDITIONAL_APPROVAL",
    ]
    assert results == expected

    # Test UNKNOWN
    df_unk = pl.DataFrame(
        {"product_number": ["P1"], "medicine_name": ["M"], "authorisation_status": ["Unknown Status"]}
    )
    assert clean_epar_bronze(df_unk)["status_normalized"][0] == "UNKNOWN"


def test_clean_epar_empty_optional_fields() -> None:
    """
    Test robustness when optional columns are missing or contain nulls.
    """
    data = {
        "product_number": ["EMEA/H/C/999"],
        "medicine_name": ["Med"],
        # Missing atc_code, active_substance
    }
    df = pl.DataFrame(data)

    # Should not crash
    cleaned = clean_epar_bronze(df)

    assert "base_procedure_id" in cleaned.columns
    # Lists should exist but be null/empty because of the else block I added
    assert "active_substance_list" in cleaned.columns
    # It should be a Series of Nulls
    assert cleaned["active_substance_list"].null_count() == 1

    # Test with Nulls in existing columns
    data_null = {
        "product_number": ["EMEA/H/C/888"],
        "medicine_name": ["Med"],
        "active_substance": [None],
        "atc_code": [None],
        "authorisation_status": [None],
    }
    df_null = pl.DataFrame(data_null)
    cleaned_null = clean_epar_bronze(df_null)

    # Nulls in string cols might become nulls after replace?
    # str.replace_all on Null returns Null.
    assert cleaned_null["active_substance"][0] is None
    # Lists should be Null (not empty list)
    assert cleaned_null["active_substance_list"][0] is None


def test_clean_epar_atc_dirty_extraction() -> None:
    """
    Complex Case: Extract ATC codes embedded in text (e.g., "A01BC01 (tablet)").
    """
    data = {
        "product_number": ["P1", "P2", "P3"],
        "medicine_name": ["M1", "M2", "M3"],
        "atc_code": [
            "A01BC01 (tablet)",  # Valid with noise
            "B02AA02; C03BB03 (syrup)",  # Mixed clean and dirty
            "Invalid Code",  # No code
        ],
        "authorisation_status": ["Authorised"] * 3,
    }
    df = pl.DataFrame(data)

    cleaned = clean_epar_bronze(df)

    # Check Row 1
    assert cleaned["atc_code_list"][0].to_list() == ["A01BC01"]

    # Check Row 2
    assert sorted(cleaned["atc_code_list"][1].to_list()) == ["B02AA02", "C03BB03"]

    # Check Row 3 (Should be empty list, not Null list, because input string was not Null)
    # Wait, my logic filters non-matches. If all filtered, it returns empty list.
    assert cleaned["atc_code_list"][2].to_list() == []
