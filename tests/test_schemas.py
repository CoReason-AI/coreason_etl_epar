# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_epar


import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from coreason_etl_epar.schemas import EPARSourceRow, SPOROrganisationRow


def test_epar_source_row_happy_path() -> None:
    row = EPARSourceRow(
        category="Human",
        product_number="EMEA/H/C/001234",
        medicine_name="SuperDrug",
        marketing_authorisation_holder="PharmaCorp",
        active_substance="Substance X",
        authorisation_status="Authorised",
        url="https://www.ema.europa.eu/en/medicines/human/EPAR/superdrug",
    )
    assert row.category == "Human"
    assert row.product_number == "EMEA/H/C/001234"
    assert row.medicine_name == "SuperDrug"
    assert row.marketing_authorisation_holder == "PharmaCorp"
    assert row.active_substance == "Substance X"
    assert row.authorisation_status == "Authorised"
    assert str(row.url) == "https://www.ema.europa.eu/en/medicines/human/EPAR/superdrug"

    # Check defaults
    assert row.therapeutic_area is None
    assert row.atc_code is None
    assert row.generic is False
    assert row.biosimilar is False
    assert row.orphan is False
    assert row.conditional_approval is False
    assert row.exceptional_circumstances is False
    assert row.revision_date is None


def test_epar_source_row_refusal_missing_optionals() -> None:
    # Testing that Refusals can miss optional fields
    row = EPARSourceRow(
        category="Human",
        product_number="EMEA/H/C/005678",
        medicine_name="FailDrug",
        marketing_authorisation_holder="FailCorp",
        active_substance="Substance Y",
        authorisation_status="Refused",
        url="https://www.ema.europa.eu/en/medicines/human/EPAR/faildrug",
    )
    assert row.product_number == "EMEA/H/C/005678"
    assert row.authorisation_status == "Refused"
    assert row.atc_code is None


def test_epar_source_row_invalid_category() -> None:
    with pytest.raises(ValidationError) as exc_info:
        EPARSourceRow(
            category="Veterinary",
            product_number="EMEA/V/C/009999",
            medicine_name="VetDrug",
            marketing_authorisation_holder="VetCorp",
            active_substance="Substance Z",
            authorisation_status="Authorised",
            url="https://www.ema.europa.eu/en/medicines/vet/EPAR/vetdrug",
        )
    assert "1 validation error for EPARSourceRow\ncategory\n  Input should be 'Human'" in str(exc_info.value)


def test_epar_source_row_invalid_product_number() -> None:
    with pytest.raises(ValidationError) as exc_info:
        EPARSourceRow(
            category="Human",
            product_number="INVALID/H/C/001234",
            medicine_name="BadIDDrug",
            marketing_authorisation_holder="PharmaCorp",
            active_substance="Substance X",
            authorisation_status="Authorised",
            url="https://www.ema.europa.eu/en/medicines/human/EPAR/baddrug",
        )
    assert "Invalid EMA Product Number format" in str(exc_info.value)


@given(  # type: ignore[misc]
    medicine_name=st.text(min_size=1),
    mah=st.text(min_size=1),
    substance=st.text(min_size=1),
    status=st.text(min_size=1),
    url=st.from_regex(r"^https://www\.example\.com/[a-z]+$", fullmatch=True),
)
def test_epar_source_row_fuzz(medicine_name: str, mah: str, substance: str, status: str, url: str) -> None:
    row = EPARSourceRow(
        category="Human",
        product_number="EMEA/H/C/123456",
        medicine_name=medicine_name,
        marketing_authorisation_holder=mah,
        active_substance=substance,
        authorisation_status=status,
        url=url,
    )
    assert row.medicine_name == medicine_name
    assert row.marketing_authorisation_holder == mah


def test_spor_organisation_row_happy_path() -> None:
    row = SPOROrganisationRow(org_id="ORG100000000", org_name="PharmaCorp Ltd")
    assert row.org_id == "ORG100000000"
    assert row.org_name == "PharmaCorp Ltd"


def test_spor_organisation_row_missing_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        SPOROrganisationRow(org_id="ORG100000000")  # type: ignore[call-arg]
    assert "1 validation error for SPOROrganisationRow\norg_name\n  Field required" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        SPOROrganisationRow(org_name="PharmaCorp Ltd")  # type: ignore[call-arg]
    assert "1 validation error for SPOROrganisationRow\norg_id\n  Field required" in str(exc_info.value)


@given(org_id=st.text(min_size=1), org_name=st.text(min_size=1))  # type: ignore[misc]
def test_spor_organisation_row_fuzz(org_id: str, org_name: str) -> None:
    row = SPOROrganisationRow(org_id=org_id, org_name=org_name)
    assert row.org_id == org_id
    assert row.org_name == org_name
