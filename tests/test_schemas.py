# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_epar


from datetime import datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from coreason_etl_epar.schemas import (
    BridgeMedicineFeatures,
    DimMedicine,
    EPARSourceRow,
    FactRegulatoryHistory,
    FeatureTypeEnum,
    RegulatoryStatusEnum,
    SPOROrganisationRow,
)


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


@given(
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


@given(org_id=st.text(min_size=1), org_name=st.text(min_size=1))
def test_spor_organisation_row_fuzz(org_id: str, org_name: str) -> None:
    row = SPOROrganisationRow(org_id=org_id, org_name=org_name)
    assert row.org_id == org_id
    assert row.org_name == org_name


def test_regulatory_status_enum() -> None:
    assert RegulatoryStatusEnum.APPROVED == "APPROVED"
    assert RegulatoryStatusEnum.CONDITIONAL_APPROVAL == "CONDITIONAL_APPROVAL"
    assert RegulatoryStatusEnum.EXCEPTIONAL_CIRCUMSTANCES == "EXCEPTIONAL_CIRCUMSTANCES"
    assert RegulatoryStatusEnum.REJECTED == "REJECTED"
    assert RegulatoryStatusEnum.WITHDRAWN == "WITHDRAWN"
    assert RegulatoryStatusEnum.SUSPENDED == "SUSPENDED"


def test_feature_type_enum() -> None:
    assert FeatureTypeEnum.ATC_CODE == "ATC_CODE"
    assert FeatureTypeEnum.SUBSTANCE == "SUBSTANCE"
    assert FeatureTypeEnum.THERAPEUTIC_AREA == "THERAPEUTIC_AREA"


def test_dim_medicine_happy_path() -> None:
    dim = DimMedicine(
        coreason_id="123e4567-e89b-12d3-a456-426614174000",
        product_number="EMEA/H/C/001234",
        medicine_name="SuperDrug",
        base_procedure_id="001234",
        brand_name="SuperBrand",
        is_biosimilar=True,
        is_generic=False,
        is_orphan=False,
        has_conditional_approval=False,
        has_exceptional_circumstances=False,
        ema_product_url="https://www.ema.europa.eu/en/medicines/human/EPAR/superdrug",
    )
    assert dim.coreason_id == "123e4567-e89b-12d3-a456-426614174000"
    assert dim.product_number == "EMEA/H/C/001234"
    assert dim.medicine_name == "SuperDrug"
    assert dim.base_procedure_id == "001234"
    assert dim.brand_name == "SuperBrand"
    assert dim.is_biosimilar is True
    assert dim.is_generic is False
    assert dim.is_orphan is False
    assert dim.has_conditional_approval is False
    assert dim.has_exceptional_circumstances is False
    assert str(dim.ema_product_url) == "https://www.ema.europa.eu/en/medicines/human/EPAR/superdrug"


def test_dim_medicine_defaults() -> None:
    dim = DimMedicine(
        coreason_id="123e4567-e89b-12d3-a456-426614174000",
        product_number="EMEA/H/C/001234",
        medicine_name="SuperDrug",
        base_procedure_id="001234",
        ema_product_url="https://www.ema.europa.eu/en/medicines/human/EPAR/superdrug",
    )
    assert dim.brand_name is None
    assert dim.is_biosimilar is False
    assert dim.is_generic is False
    assert dim.is_orphan is False
    assert dim.has_conditional_approval is False
    assert dim.has_exceptional_circumstances is False


def test_fact_regulatory_history_happy_path() -> None:
    dt_from = datetime(2023, 1, 1)
    dt_to = datetime(2024, 1, 1)
    fact = FactRegulatoryHistory(
        history_id="hist-123",
        coreason_id="123e4567-e89b-12d3-a456-426614174000",
        status=RegulatoryStatusEnum.APPROVED,
        valid_from=dt_from,
        valid_to=dt_to,
        is_current=False,
        marketing_authorisation_holder="PharmaCorp",
        spor_mah_id="ORG1000",
        org_name="pharmacorp",
    )
    assert fact.history_id == "hist-123"
    assert fact.coreason_id == "123e4567-e89b-12d3-a456-426614174000"
    assert fact.status == RegulatoryStatusEnum.APPROVED
    assert fact.valid_from == dt_from
    assert fact.valid_to == dt_to
    assert fact.is_current is False
    assert fact.marketing_authorisation_holder == "PharmaCorp"
    assert fact.spor_mah_id == "ORG1000"
    assert fact.org_name == "pharmacorp"


def test_fact_regulatory_history_defaults() -> None:
    dt_from = datetime(2023, 1, 1)
    fact = FactRegulatoryHistory(
        history_id="hist-123",
        coreason_id="123e4567-e89b-12d3-a456-426614174000",
        status=RegulatoryStatusEnum.APPROVED,
        valid_from=dt_from,
        is_current=True,
    )
    assert fact.valid_to is None
    assert fact.marketing_authorisation_holder is None
    assert fact.spor_mah_id is None
    assert fact.org_name is None


def test_bridge_medicine_features_happy_path() -> None:
    bridge = BridgeMedicineFeatures(
        coreason_id="123e4567-e89b-12d3-a456-426614174000",
        feature_type=FeatureTypeEnum.ATC_CODE,
        feature_value="A10BA02",
    )
    assert bridge.coreason_id == "123e4567-e89b-12d3-a456-426614174000"
    assert bridge.feature_type == FeatureTypeEnum.ATC_CODE
    assert bridge.feature_value == "A10BA02"


@given(
    coreason_id=st.text(min_size=1),
    product_number=st.text(min_size=1),
    medicine_name=st.text(min_size=1),
    base_procedure_id=st.text(min_size=1),
    url=st.from_regex(r"^https://www\.example\.com/[a-z]+$", fullmatch=True),
)
def test_dim_medicine_fuzz(
    coreason_id: str, product_number: str, medicine_name: str, base_procedure_id: str, url: str
) -> None:
    dim = DimMedicine(
        coreason_id=coreason_id,
        product_number=product_number,
        medicine_name=medicine_name,
        base_procedure_id=base_procedure_id,
        ema_product_url=url,
    )
    assert dim.coreason_id == coreason_id
    assert dim.product_number == product_number
    assert dim.medicine_name == medicine_name
    assert dim.base_procedure_id == base_procedure_id
