from pathlib import Path

import pandas as pd
import pytest

from coreason_etl_epar.ingest import epar_index


@pytest.fixture  # type: ignore[misc]
def dummy_excel_file(tmp_path: Path) -> str:
    file_path = tmp_path / "medicines.xlsx"

    # Create a DataFrame with some valid and invalid data
    data = {
        "Category": ["Human", "Veterinary", "Human", "Human"],
        "Product number": ["EMEA/H/C/001234", "EMEA/V/C/009999", "INVALID_NUM", "EMEA/H/C/005678"],
        "Medicine name": ["Med A", "Med Vet", "Med B", "Med C"],
        "Marketing authorisation holder": ["Holder A", "Holder V", "Holder B", "Holder C"],
        "Active substance": ["Sub A", "Sub V", "Sub B", "Sub C"],
        "Therapeutic area": ["Area A", "Area V", "Area B", "Area C"],
        "ATC code": ["A01", "V01", "B02", "C03"],
        "Generic": [False, False, False, True],
        "Biosimilar": [False, False, False, False],
        "Orphan": [False, False, False, False],
        "Conditional approval": [False, False, False, False],
        "Exceptional circumstances": [False, False, False, False],
        "Authorisation status": ["Authorised", "Authorised", "Authorised", "Refused"],
        "Revision date": [None, None, None, None],
        "URL": ["http://a.com", "http://v.com", "http://b.com", "http://c.com"],
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
    # Row 0: Human, Valid -> Should be yielded
    # Row 1: Veterinary -> Should be filtered out
    # Row 2: Human, Invalid Product Number -> Validation error, log warning, skip (based on current impl)
    # Row 3: Human, Valid Refused -> Should be yielded

    assert len(rows) == 2

    row1 = rows[0]
    assert row1["product_number"] == "EMEA/H/C/001234"
    assert row1["category"] == "Human"

    row2 = rows[1]
    assert row2["product_number"] == "EMEA/H/C/005678"
    assert row2["medicine_name"] == "Med C"


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
