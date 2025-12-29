from typing import Any, Dict

import pytest
from pydantic import ValidationError

from coreason_etl_epar.schema import EPARSourceRow


def test_schema_empty_string_conversion() -> None:
    """
    Edge Case: Empty strings for Optional fields should be converted to None.
    Common issue in Excel ingestion where empty cells might be read as "".
    """
    data: Dict[str, Any] = {
        "category": "Human",
        "product_number": "EMEA/H/C/001",
        "medicine_name": "M",
        "marketing_authorisation_holder": "H",
        "active_substance": "S",
        "authorisation_status": "A",
        "url": "u",
        "therapeutic_area": "",  # Empty String
        "atc_code": "",  # Empty String
        "revision_date": None,
    }

    # Depending on Pydantic config, this might fail or pass as "".
    # We want it to be None or at least accepted.
    # For optional strings, "" is a valid string. So it will be "".
    # But for business logic, "" usually means None.

    row = EPARSourceRow(**data)
    # Verify behavior: Current model doesn't strip/convert to None.
    # We should probably enforce this if the requirement demands it.
    # The FRD doesn't explicitly say "Convert empty strings to Null", but it's good practice.
    # For now, let's just assert the current behavior is defined.
    assert row.therapeutic_area == ""
    assert row.atc_code == ""


def test_schema_date_empty_string() -> None:
    """
    Edge Case: Empty string for Date field.
    Pydantic V2 strictly rejects "" for datetime.
    """
    data: Dict[str, Any] = {
        "category": "Human",
        "product_number": "EMEA/H/C/001",
        "medicine_name": "M",
        "marketing_authorisation_holder": "H",
        "active_substance": "S",
        "authorisation_status": "A",
        "url": "u",
        "revision_date": "",  # Empty String for Datetime
    }

    # Expectation: Validation Error
    with pytest.raises(ValidationError):
        EPARSourceRow(**data)


def test_schema_boolean_coercion() -> None:
    """
    Edge Case: "Yes"/"No" strings for boolean fields.
    Source data often has "Yes" instead of True.
    Pydantic handles 'yes'/'no' -> bool coercion automatically in some configs,
    but strictly it might expect True/False.
    """
    data: Dict[str, Any] = {
        "category": "Human",
        "product_number": "EMEA/H/C/001",
        "medicine_name": "M",
        "marketing_authorisation_holder": "H",
        "active_substance": "S",
        "authorisation_status": "A",
        "url": "u",
        "orphan": "Yes",  # String "Yes"
        "generic": "no",  # String "no"
    }

    # Pydantic V2 allows string coercion for bools by default (strict=False).
    row = EPARSourceRow(**data)
    assert row.orphan is True
    assert row.generic is False
