import polars as pl

from coreason_etl_epar.transform_enrich import enrich_epar, jaro_winkler
from coreason_etl_epar.transform_silver import clean_epar_bronze


def test_jaro_winkler() -> None:
    # Exact match
    assert jaro_winkler("test", "test") == 1.0
    # No match
    assert jaro_winkler("abc", "xyz") == 0.0
    # Empty strings
    assert jaro_winkler("", "abc") == 0.0
    assert jaro_winkler("abc", "") == 0.0
    assert jaro_winkler("", "") == 1.0  # Handled by first check
    # Branch check for len == 0 handled.
    # Check transposition
    # "dixon" vs "dicksonx"
    # Matches: d, i, o, n (4).
    # Transposition logic triggered if s1[i] != s2[k] but matched.
    # 'martha' vs 'marhta' -> t and h are swapped.
    assert jaro_winkler("martha", "marhta") > 0.9
    # Check no match branch inside inner loop (lines 48-51) is implicitly covered by low scores.

    # Check break in prefix loop
    assert jaro_winkler("abc", "abd") < 1.0  # prefix = 2
    assert jaro_winkler("dixon", "dicksonx") > 0.7

    # Check duplicate char handling (covers 'continue' when s2_matches[j] is True)
    # s1="aa", s2="a"
    # i=0 matches s2[0]. s2_matches[0]=True.
    # i=1 sees s2[0] used -> continue.
    assert jaro_winkler("aa", "a") > 0.0

    # Specific case for hitting 'continue' in inner loop
    # s1="aba", s2="bab"
    # i=0 ('a') matches s2[1]. s2_matches[1]=True
    # i=1 ('b') matches s2[0]. s2_matches[0]=True
    # i=2 ('a') looks at s2[0] (used->continue? No s2[0] is 'b'!=s1[2])
    # Let's construct a case: s1="aa", s2="aa".
    # i=0 ('a') matches s2[0]. s2_matches[0]=True.
    # i=1 ('a') checks s2[0]. s2_matches[0] is True -> continue (Hits line 15).
    # Then checks s2[1]. Match.
    assert jaro_winkler("aa", "aa") == 1.0

    # Extra cases to force coverage
    assert jaro_winkler("CRATE", "TRACE") > 0.0  # Transposition
    assert jaro_winkler("DwAyNE", "DuANE") > 0.0
    assert jaro_winkler("ABC", "XBC") > 0.0  # Match logic


def test_status_normalization_in_enrich_context() -> None:
    # Ensure clean_epar_bronze is used for status normalization validation
    # This was previously test_normalize_status but now we test the integration
    data = {
        "product_number": ["P1"],
        "medicine_name": ["M1"],
        "authorisation_status": ["Authorised"],
    }
    df = pl.DataFrame(data)
    cleaned = clean_epar_bronze(df)
    assert cleaned["status_normalized"][0] == "APPROVED"


def test_enrich_epar_logic() -> None:
    # Setup Data
    df = pl.DataFrame(
        {
            "product_number": ["EMEA/H/C/001234", "EMEA/H/C/999"],
            "active_substance": ["Sub A / Sub B", "Sub C + Sub D"],
            "atc_code": ["A01BC01;B02AA02", "C03BB03,D04CC04"],
            "authorisation_status": ["Authorised", "Refused"],
            "marketing_authorisation_holder": ["Pharma Corp", "BioTech Inc"],
        }
    )

    # Clean the DataFrame first (Pipeline Simulation)
    cleaned_df = clean_epar_bronze(df)

    # Use names that are very close to satisfy > 0.90 threshold
    spor_df = pl.DataFrame(
        {"name": ["Pharma Corp.", "BioTech Inc.", "Other Co"], "org_id": ["ORG-100", "ORG-200", "ORG-300"]}
    )

    # Run Enrichment
    result = enrich_epar(cleaned_df, spor_df)

    # Check Base Procedure ID (comes from clean_epar_bronze)
    assert result.filter(pl.col("product_number") == "EMEA/H/C/001234")["base_procedure_id"].item() == "001234"

    # Check Substance List (comes from clean_epar_bronze)
    row1 = result.row(0, named=True)
    assert row1["active_substance_list"] == ["Sub A", "Sub B"]
    row2 = result.row(1, named=True)
    assert row2["active_substance_list"] == ["Sub C", "Sub D"]  # Handled +

    # Check ATC List (comes from clean_epar_bronze)
    assert row1["atc_code_list"] == ["A01BC01", "B02AA02"]
    assert row2["atc_code_list"] == ["C03BB03", "D04CC04"]  # Handled ,

    # Check Status
    assert row1["status_normalized"] == "APPROVED"

    # Check Fuzzy Match
    # Pharma Corp vs Pharma Corp. -> Should be > 0.90

    assert row1["spor_mah_id"] == "ORG-100"
    assert row2["spor_mah_id"] == "ORG-200"


def test_enrich_epar_no_match() -> None:
    df = pl.DataFrame(
        {
            "product_number": ["EMEA/H/C/001234"],
            "active_substance": ["A"],
            "atc_code": ["A01BC01"],
            "authorisation_status": ["Authorised"],
            "marketing_authorisation_holder": ["Unique Name"],
        }
    )

    cleaned_df = clean_epar_bronze(df)

    spor_df = pl.DataFrame({"name": ["Totally Different"], "org_id": ["ORG-999"]})

    result = enrich_epar(cleaned_df, spor_df)
    assert result["spor_mah_id"].item() is None


def test_enrich_tie_breaker_determinism() -> None:
    # Setup: 1 EPAR MAH, 2 SPOR MAHs with IDENTICAL names (score 1.0) but different IDs.
    # We want deterministic choice (lowest ID).

    df = pl.DataFrame(
        {
            "product_number": ["P1"],
            "medicine_name": ["M1"],
            "marketing_authorisation_holder": ["Pharma Corp"],
            "active_substance": ["S1"],
            "atc_code": ["A01BC01"],
            "authorisation_status": ["Authorised"],
            "url": ["u"],
        }
    )

    # We call clean even if not needed for this specific test to ensure schema consistency if needed
    cleaned_df = clean_epar_bronze(df)

    # SPOR data with duplicates or very similar
    spor_df = pl.DataFrame(
        {
            "name": ["Pharma Corp", "Pharma Corp"],
            "org_id": ["ORG-002", "ORG-001"],  # ORG-001 should be picked if sorting by ID ascending
        }
    )

    result = enrich_epar(cleaned_df, spor_df)

    # Assert
    assert result["spor_mah_id"].item() == "ORG-001"


def test_unicode_handling() -> None:
    # Enrichment (Jaro-Winkler with unicode)
    score = jaro_winkler("Société", "Societe")
    assert score > 0.0


def test_enrich_epar_empty_spor() -> None:
    df = pl.DataFrame(
        {
            "product_number": ["EMEA/H/C/001234"],
            "active_substance": ["A"],
            "atc_code": ["A01BC01"],
            "authorisation_status": ["Authorised"],
            "marketing_authorisation_holder": ["Unique Name"],
        }
    )

    cleaned_df = clean_epar_bronze(df)

    spor_df = pl.DataFrame(schema={"name": pl.String, "org_id": pl.String})

    result = enrich_epar(cleaned_df, spor_df)
    assert result["spor_mah_id"].item() is None


def test_atc_code_validation() -> None:
    # Test strict L7 validation (Regex: ^[A-Z]\d{2}[A-Z]{2}\d{2}$)
    # Now this logic resides in clean_epar_bronze
    atc_codes = [
        "A01BC01",  # Valid
        "X01",  # Too short
        "A01BC012",  # Too long
        "101BC01",  # Starts with digit
        "A011234",  # Missing middle letters
        "A01BC01;X01",  # One valid, one invalid
        None,  # Null
    ]
    length = len(atc_codes)

    # Use unique product numbers to ensure stable sorting
    product_numbers = [f"P{i}" for i in range(length)]

    df = pl.DataFrame(
        {
            "product_number": product_numbers,
            "active_substance": ["S1"] * length,
            "atc_code": atc_codes,
            "authorisation_status": ["A"] * length,
            "marketing_authorisation_holder": ["H"] * length,
        }
    )

    # Run cleaning
    result = clean_epar_bronze(df)

    atc_lists = result["atc_code_list"].to_list()

    assert atc_lists[0] == ["A01BC01"]  # Valid kept
    assert atc_lists[1] == []  # Too short dropped
    assert atc_lists[2] == []  # Too long dropped
    assert atc_lists[3] == []  # Wrong start dropped
    assert atc_lists[4] == []  # Wrong format dropped
    assert atc_lists[5] == ["A01BC01"]  # Mixed: keep valid, drop invalid
    assert atc_lists[6] is None  # Null stays None


def test_jaro_winkler_unicode() -> None:
    s1 = "café"
    s2 = "cafe"
    score = jaro_winkler(s1, s2)
    assert 0.8 < score < 1.0

    s_nfd = "cafe\u0301"
    s_nfc = "caf\u00e9"
    assert s_nfd != s_nfc
    score_norm = jaro_winkler(s_nfd, s_nfc)
    assert score_norm > 0.8
