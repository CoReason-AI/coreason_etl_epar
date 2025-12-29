from datetime import datetime

import pytest
from coreason_etl_epar.schema import EPARSourceRow
from pydantic import ValidationError


def test_valid_epar_source_row() -> None:
    data = {
        "category": "Human",
        "product_number": "EMEA/H/C/001234",
        "medicine_name": "Test Medicine",
        "marketing_authorisation_holder": "Test MAH",
        "active_substance": "Test Substance",
        "therapeutic_area": "Test Area",
        "atc_code": "A01",
        "generic": True,
        "biosimilar": False,
        "orphan": False,
        "conditional_approval": False,
        "exceptional_circumstances": False,
        "authorisation_status": "Authorised",
        "revision_date": datetime.now(),
        "url": "http://example.com",
    }
    row = EPARSourceRow(**data)
    assert row.product_number == "EMEA/H/C/001234"
    assert row.category == "Human"


def test_optional_fields() -> None:
    data = {
        "category": "Human",
        "product_number": "EMEA/H/C/005678",
        "medicine_name": "Refused Medicine",
        "marketing_authorisation_holder": "Test MAH",
        "active_substance": "Test Substance",
        # Missing therapeutic_area and atc_code
        "authorisation_status": "Refused",
        "url": "http://example.com",
    }
    row = EPARSourceRow(**data)
    assert row.therapeutic_area is None
    assert row.atc_code is None
    assert row.generic is False  # Default value


def test_invalid_category() -> None:
    data = {
        "category": "Veterinary",
        "product_number": "EMEA/V/C/001234",
        "medicine_name": "Vet Medicine",
        "marketing_authorisation_holder": "Vet MAH",
        "active_substance": "Vet Substance",
        "authorisation_status": "Authorised",
        "url": "http://example.com",
    }
    with pytest.raises(ValidationError) as excinfo:
        EPARSourceRow(**data)
    assert "Input should be 'Human'" in str(excinfo.value)


def test_invalid_product_number_format() -> None:
    data = {
        "category": "Human",
        "product_number": "INVALID/123",
        "medicine_name": "Test Medicine",
        "marketing_authorisation_holder": "Test MAH",
        "active_substance": "Test Substance",
        "authorisation_status": "Authorised",
        "url": "http://example.com",
    }
    with pytest.raises(ValidationError) as excinfo:
        EPARSourceRow(**data)
    assert "Invalid EMA Product Number format" in str(excinfo.value)


def test_pydantic_invalid_date_format() -> None:
    # Source provides date as "2024/01/01" but we expect it to be parsed.
    # Pydantic usually handles standard ISO. If dlt provides strings, Pydantic might coerce.
    # But if format is completely wrong "NotADate", it should fail.

    data = {
        "category": "Human",
        "product_number": "EMEA/H/C/001",
        "medicine_name": "M",
        "marketing_authorisation_holder": "H",
        "active_substance": "S",
        "authorisation_status": "A",
        "url": "u",
        "revision_date": "NotADate",  # Invalid
    }

    with pytest.raises(ValidationError):
        EPARSourceRow(**data)
