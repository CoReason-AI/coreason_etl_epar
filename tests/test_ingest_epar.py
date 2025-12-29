from pathlib import Path

import pandas as pd
import pytest

from coreason_etl_epar.ingest import epar_index


@pytest.fixture  # type: ignore[misc]
def dummy_excel_file(tmp_path: Path) -> str:
    file_path = tmp_path / "medicines.xlsx"

    # Create a DataFrame with some valid and invalid data
    # Added mixed casing for Category to test case-insensitivity
    # Added whitespace test (" Human ")
    # Added missing Product Number case
    data = {
        "Category": ["Human", "Veterinary", "Human", "HUMAN", "human", "VETERINARY", " Human ", "Human"],
        "Product number": [
            "EMEA/H/C/001234",
            "EMEA/V/C/009999",
            "INVALID_NUM",
            "EMEA/H/C/005678",
            "EMEA/H/C/000001",
            "EMEA/V/C/008888",
            "EMEA/H/C/009000",
            None,  # Missing Product Number
        ],
        "Medicine name": ["Med A", "Med Vet", "Med B", "Med C", "Med D", "Med V2", "Med Pad", "Med NoID"],
        "Marketing authorisation holder": [
            "Holder A",
            "Holder V",
            "Holder B",
            "Holder C",
            "Holder D",
            "Holder V2",
            "Holder P",
            "Holder N",
        ],
        "Active substance": ["Sub A", "Sub V", "Sub B", "Sub C", "Sub D", "Sub V2", "Sub P", "Sub N"],
        "Therapeutic area": ["Area A", "Area V", "Area B", "Area C", "Area D", "Area V2", "Area P", "Area N"],
        "ATC code": ["A01", "V01", "B02", "C03", "D04", "V02", "P05", "N06"],
        "Generic": [False, False, False, True, False, False, False, False],
        "Biosimilar": [False, False, False, False, False, False, False, False],
        "Orphan": [False, False, False, False, False, False, False, False],
        "Conditional approval": [False, False, False, False, False, False, False, False],
        "Exceptional circumstances": [False, False, False, False, False, False, False, False],
        "Authorisation status": [
            "Authorised",
            "Authorised",
            "Authorised",
            "Refused",
            "Authorised",
            "Authorised",
            "Authorised",
            "Authorised",
        ],
        "Revision date": [None, None, None, None, None, None, None, None],
        "URL": [
            "http://a.com",
            "http://v.com",
            "http://b.com",
            "http://c.com",
            "http://d.com",
            "http://v2.com",
            "http://p.com",
            "http://n.com",
        ],
    }

    # Create Excel file using pandas (requires openpyxl)
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)

    return str(file_path)


def test_epar_index_resource(dummy_excel_file: str) -> None:
    # Iterate over the resource
    resource = epar_index(dummy_excel_file)
    rows = list(resource)

    # Analysis of expected output:
    # Row 0: Human, Valid -> Yielded
    # Row 1: Veterinary -> Filtered
    # Row 2: Human, Invalid Product Number -> Quarantine
    # Row 3: HUMAN, Valid Refused -> Yielded
    # Row 4: human, Valid -> Yielded
    # Row 5: VETERINARY -> Filtered
    # Row 6: " Human ", Valid (after strip) -> Yielded
    # Row 7: Human, Missing Product Number -> Quarantine

    # Total yielded: 4 valid + 2 quarantine = 6
    # Valid: Row 0, Row 3, Row 4, Row 6.
    # Quarantine: Row 2, Row 7.

    # Because we now yield quarantine items, the length should be 6
    assert len(rows) == 6

    valid_rows = [r for r in rows if "error_message" not in r]
    quarantine_rows = [r for r in rows if "error_message" in r]

    assert len(valid_rows) == 4
    assert len(quarantine_rows) == 2

    # Check case insensitivity & stripping
    product_numbers = [r["product_number"] for r in valid_rows]
    assert "EMEA/H/C/001234" in product_numbers  # Human
    assert "EMEA/H/C/005678" in product_numbers  # HUMAN
    assert "EMEA/H/C/000001" in product_numbers  # human
    assert "EMEA/H/C/009000" in product_numbers  # " Human "

    # Ensure veterinary ones are not there
    assert "EMEA/V/C/009999" not in product_numbers
    assert "EMEA/V/C/008888" not in product_numbers

    # Check Quarantine Records
    # 1. Invalid Number
    q_invalid = next(r for r in quarantine_rows if r["product_number"] == "INVALID_NUM")
    assert "Invalid EMA Product Number format" in q_invalid["error_message"]
    assert "ingestion_ts" in q_invalid
    assert q_invalid["raw_data"]["product_number"] == "INVALID_NUM"

    # 2. Missing Number
    # Note: When product_number is None/NaN, our ingest code defaults it to "UNKNOWN" in the log/quarantine struct key
    # but the raw_data should contain the original None (or Nan/Null from polars).
    q_missing = next(r for r in quarantine_rows if r["product_number"] == "UNKNOWN")
    assert "product_number" in q_missing["error_message"]  # Pydantic error for missing field
    assert q_missing["raw_data"]["medicine_name"] == "Med NoID"


def test_ingest_file_not_found() -> None:
    # Blind exception check fixed
    # We expect a specific error if possible, but dlt/polars might raise different ones.
    # Polars raises FileNotFoundError or ComputeError.
    # dlt might wrap it.
    # The previous ruff error was B017: Do not assert blind exception: `Exception`.
    # I should use `pytest.raises(Exception)` but inspect the message, OR catch specific exceptions.
    # But `polars` exceptions are custom.
    # I'll try to assert specific message part if I catch Exception.

    with pytest.raises(Exception) as excinfo:
        list(epar_index("non_existent_file.xlsx"))

    # Check if message contains something relevant
    msg = str(excinfo.value)
    assert "No such file" in msg or "does not exist" in msg or "no workbook found" in msg


def test_ingest_missing_category_column(tmp_path: Path) -> None:
    file_path = tmp_path / "bad_columns.xlsx"
    df = pd.DataFrame({"Wrong Col": [1, 2, 3]})
    df.to_excel(file_path, index=False)

    # Should log error and return empty
    resource = epar_index(str(file_path))
    rows = list(resource)
    assert len(rows) == 0
