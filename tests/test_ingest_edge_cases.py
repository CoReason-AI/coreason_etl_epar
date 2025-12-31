from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from coreason_etl_epar.ingest import epar_index


def write_excel_pandas(data: Dict[str, Any], path: str) -> None:
    # Use pandas directly to write excel via openpyxl (installed)
    # Avoids Polars -> Pandas conversion requiring pyarrow
    pd.DataFrame(data).to_excel(path, index=False)


def test_ingest_empty_string_optional_fields(tmp_path: Path) -> None:
    """
    Test that empty strings in optional fields (atc_code, therapeutic_area)
    are converted to None and pass validation.
    """
    data = {
        "Category": ["Human"],
        "Product number": ["EMEA/H/C/001"],
        "Medicine name": ["Med A"],
        "Marketing authorisation holder": ["Holder A"],
        "Active substance": ["Sub A"],
        "Authorisation status": ["Authorised"],
        "URL": ["http://url"],
        "ATC code": [""],  # Empty string
        "Therapeutic area": [""],  # Empty string
    }
    file_path = tmp_path / "test_empty_opt.xlsx"
    write_excel_pandas(data, str(file_path))

    # Run ingest
    results = list(epar_index(str(file_path)))

    # Expect 1 success record, 0 quarantine
    assert len(results) == 1
    record = results[0]

    # If validation fails, error_message is present
    if "error_message" in record:
        pytest.fail(f"Validation failed: {record['error_message']}")

    assert record["atc_code"] is None
    assert record["therapeutic_area"] is None


def test_ingest_empty_string_required_fields(tmp_path: Path) -> None:
    """
    Test that empty strings in required fields (Medicine name) trigger Quarantine.
    """
    data = {
        "Category": ["Human"],
        "Product number": ["EMEA/H/C/002"],
        "Medicine name": [""],  # Required, Empty
        "Marketing authorisation holder": ["Holder A"],
        "Active substance": ["Sub A"],
        "Authorisation status": ["Authorised"],
        "URL": ["http://url"],
    }
    file_path = tmp_path / "test_empty_req.xlsx"
    write_excel_pandas(data, str(file_path))

    results = list(epar_index(str(file_path)))

    assert len(results) == 1
    record = results[0]

    assert "error_message" in record
    assert record["product_number"] == "EMEA/H/C/002"
    assert (
        "Field required" in record["error_message"]
        or "String should have at least" in record["error_message"]
        or "Input should be a valid string" in record["error_message"]
    )


def test_ingest_malformed_date(tmp_path: Path) -> None:
    """
    Test that malformed dates trigger Quarantine.
    """
    data = {
        "Category": ["Human"],
        "Product number": ["EMEA/H/C/003"],
        "Medicine name": ["Med C"],
        "Marketing authorisation holder": ["Holder A"],
        "Active substance": ["Sub A"],
        "Authorisation status": ["Authorised"],
        "URL": ["http://url"],
        "Revision date": ["Not A Date"],  # Malformed
    }
    file_path = tmp_path / "test_bad_date.xlsx"
    write_excel_pandas(data, str(file_path))

    results = list(epar_index(str(file_path)))

    assert len(results) == 1
    record = results[0]
    assert "error_message" in record
    assert (
        "Input should be a valid datetime" in record["error_message"]
        or "invalid datetime" in str(record["error_message"]).lower()
    )


def test_ingest_empty_date(tmp_path: Path) -> None:
    """
    Test that empty string date is treated as None (Success), NOT Quarantine.
    """
    data = {
        "Category": ["Human"],
        "Product number": ["EMEA/H/C/003b"],
        "Medicine name": ["Med Cb"],
        "Marketing authorisation holder": ["Holder A"],
        "Active substance": ["Sub A"],
        "Authorisation status": ["Authorised"],
        "URL": ["http://url"],
        "Revision date": [""],  # Empty string
    }
    file_path = tmp_path / "test_empty_date.xlsx"
    write_excel_pandas(data, str(file_path))

    results = list(epar_index(str(file_path)))

    assert len(results) == 1
    record = results[0]

    # Should NOT be quarantine
    if "error_message" in record:
        pytest.fail(f"Validation failed for empty date: {record['error_message']}")

    assert record["revision_date"] is None


def test_ingest_missing_product_number(tmp_path: Path) -> None:
    """
    Test that missing Product Number handles gracefully (Quarantine with UNKNOWN).
    """
    # Type annotation for None in List
    data: Dict[str, Any] = {
        "Category": ["Human"],
        "Product number": [None],  # Missing
        "Medicine name": ["Med D"],
        "Marketing authorisation holder": ["Holder A"],
        "Active substance": ["Sub A"],
        "Authorisation status": ["Authorised"],
        "URL": ["http://url"],
    }
    file_path = tmp_path / "test_missing_pk.xlsx"
    write_excel_pandas(data, str(file_path))

    results = list(epar_index(str(file_path)))

    assert len(results) == 1
    record = results[0]
    assert "error_message" in record
    assert record["product_number"] == "UNKNOWN"


def test_ingest_boolean_coercion(tmp_path: Path) -> None:
    """
    Test that "Yes"/"No" strings are coerced to Boolean.
    """
    data: Dict[str, Any] = {
        "Category": ["Human"],
        "Product number": ["EMEA/H/C/005"],
        "Medicine name": ["Med E"],
        "Marketing authorisation holder": ["Holder A"],
        "Active substance": ["Sub A"],
        "Authorisation status": ["Authorised"],
        "URL": ["http://url"],
        "Orphan": ["Yes"],
        "Generic": ["no"],
        "Biosimilar": [True],  # Already bool
    }
    file_path = tmp_path / "test_bools.xlsx"
    write_excel_pandas(data, str(file_path))

    results = list(epar_index(str(file_path)))

    assert len(results) == 1
    record = results[0]
    assert "error_message" not in record

    assert record["orphan"] is True
    assert record["generic"] is False
    assert record["biosimilar"] is True
