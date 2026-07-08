# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_epar

import io
import zipfile
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from coreason_etl_epar.ingest import get_epar_index_resource, get_spor_organisations_resource


@pytest.fixture
def mock_excel_data() -> bytes:
    # Create a simple Excel file in memory
    df = pd.DataFrame(
        {
            "Category": ["Human", "Veterinary", "Human", "Human"],
            "Product number": ["EMEA/H/C/001234", "EMEA/V/C/005678", "INVALID/ID", "EMEA/H/C/009999"],
            "Medicine name": ["SuperDrug", "VetDrug", "BadIDDrug", "FailDrug"],
            "Marketing authorisation holder/company name": ["PharmaCorp", "VetCorp", "BadCorp", "FailCorp"],
            "International non-proprietary name (INN) / common name": [
                "Substance X",
                "Substance Y",
                "Substance Z",
                "Substance W",
            ],
            "Authorisation status": ["Authorised", "Authorised", "Authorised", "Refused"],
            "URL": [
                "https://www.ema.europa.eu/en/medicines/human/EPAR/superdrug",
                "https://www.ema.europa.eu/en/medicines/vet/EPAR/vetdrug",
                "https://www.ema.europa.eu/en/medicines/human/EPAR/baddrug",
                "not-a-url",
            ],
            "Generic": ["No", "No", "No", "Yes"],
            "Biosimilar": ["Yes", "No", "No", "No"],
            "Orphan medicine": ["false", "false", "false", "true"],
        }
    )
    # The source file has 8 rows of headers before the actual data
    dummy_header = pd.DataFrame([[""] * 10] * 8)

    # Write to BytesIO
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dummy_header.to_excel(writer, index=False, header=False)
        df.to_excel(writer, index=False, header=True, startrow=8)

    return output.getvalue()


@pytest.fixture
def mock_excel_data_no_category() -> bytes:
    df = pd.DataFrame(
        {
            "Product number": ["EMEA/H/C/001234"],
            "Medicine name": ["SuperDrug"],
            "Marketing authorisation holder/company name": ["PharmaCorp"],
            "International non-proprietary name (INN) / common name": ["Substance X"],
            "Authorisation status": ["Authorised"],
            "URL": ["https://www.ema.europa.eu/en/medicines/human/EPAR/superdrug"],
        }
    )
    dummy_header = pd.DataFrame([[""] * 6] * 8)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dummy_header.to_excel(writer, index=False, header=False)
        df.to_excel(writer, index=False, header=True, startrow=8)
    return output.getvalue()


@patch("requests.get")
def test_get_epar_index_resource_success(mock_get: Mock, mock_excel_data: bytes) -> None:
    # Setup mock response
    mock_response = Mock()
    mock_response.content = mock_excel_data
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    resource = get_epar_index_resource("http://fake-url.com")
    items = list(resource)

    # We expect:
    # 1. SuperDrug: Valid Human record -> emitted as dict
    # 2. VetDrug: Veterinary record -> filtered out completely
    # 3. BadIDDrug: Invalid product number -> emitted as DataItemWithMeta to quarantine
    # 4. FailDrug: Invalid URL -> emitted as DataItemWithMeta to quarantine

    # Wait, the second record is VetDrug (filtered)
    # The third is BadIDDrug (Invalid product number, quarantine)
    # The fourth is FailDrug (Invalid URL, quarantine)

    assert len(items) == 3

    # Check valid row
    valid_row = items[0]
    assert isinstance(valid_row, dict)
    # Note: data in valid_row is dumped from Pydantic, so values match EPARSourceRow
    assert valid_row["category"] == "Human"
    assert valid_row["product_number"] == "EMEA/H/C/001234"
    assert valid_row["medicine_name"] == "SuperDrug"
    assert valid_row["active_substance"] == "Substance X"
    assert valid_row["generic"] is False
    assert valid_row["biosimilar"] is True
    assert valid_row["orphan"] is False

    # Check quarantined rows
    quarantine_1 = items[1]
    assert isinstance(quarantine_1, dict)
    # In some dlt versions, metadata is on __wrapped__ or similar, or it's a subclass of dict.
    # Actually, let's just assert the error message is present since we know it yielded the quarantine dict.
    assert "error" in quarantine_1
    assert "Invalid EMA Product Number format" in quarantine_1["error"]

    quarantine_2 = items[2]
    assert isinstance(quarantine_2, dict)
    assert "error" in quarantine_2
    assert "url" in quarantine_2["error"]  # URL validation error


@patch("requests.get")
def test_get_epar_index_resource_no_category(mock_get: Mock, mock_excel_data_no_category: bytes) -> None:
    mock_response = Mock()
    mock_response.content = mock_excel_data_no_category
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    resource = get_epar_index_resource("http://fake-url.com")
    items = list(resource)

    assert len(items) == 1
    valid_row = items[0]
    assert isinstance(valid_row, dict)
    assert valid_row["category"] == "Human"  # Added as default when missing
    assert valid_row["product_number"] == "EMEA/H/C/001234"


@patch("requests.get")
def test_get_epar_index_resource_datetime_and_na(mock_get: Mock) -> None:
    import datetime

    # Test pandas NA and datetime logic
    df = pd.DataFrame(
        {
            "Product number": ["EMEA/H/C/001234"],
            "Medicine name": ["SuperDrug"],
            "Marketing authorisation holder/company name": ["PharmaCorp"],
            "International non-proprietary name (INN) / common name": ["Substance X"],
            "Authorisation status": ["Authorised"],
            "URL": ["https://www.ema.europa.eu/en/medicines/human/EPAR/superdrug"],
            "Revision date": [datetime.datetime(2023, 1, 1)],
            "Generic": [pd.NA],
            "Exceptional circumstances": ["Maybe"],
        }
    )
    dummy_header = pd.DataFrame([[""] * 9] * 8)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dummy_header.to_excel(writer, index=False, header=False)
        df.to_excel(writer, index=False, header=True, startrow=8)

    mock_response = Mock()
    mock_response.content = output.getvalue()
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    resource = get_epar_index_resource("http://fake-url.com")
    items = list(resource)

    assert len(items) == 1
    valid_row = items[0]
    assert isinstance(valid_row, dict)
    assert valid_row["revision_date"].isoformat().startswith("2023-01-01")
    assert valid_row["generic"] is False
    assert valid_row["exceptional_circumstances"] is False


@pytest.fixture
def mock_spor_xml_data() -> bytes:
    # Minimal XML structure that mimics OMS SPOR export
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <Organisations>
        <Organisation>
            <OrganisationId>ORG100000000</OrganisationId>
            <OrganisationName>Valid MAH Pharma</OrganisationName>
            <IndustrySector>Marketing Authorisation Holder</IndustrySector>
        </Organisation>
        <Organisation>
            <OrganisationId>ORG100000001</OrganisationId>
            <OrganisationName>Another Company</OrganisationName>
            <IndustrySector>Manufacturer</IndustrySector>
        </Organisation>
        <Organisation>
            <OrganisationId>ORG100000002</OrganisationId>
            <OrganisationName>Second MAH Corp</OrganisationName>
            <IndustrySector>Marketing Authorisation Holder</IndustrySector>
        </Organisation>
        <Organisation>
            <!-- Missing name but is MAH -->
            <OrganisationId>ORG100000003</OrganisationId>
            <IndustrySector>Marketing Authorisation Holder</IndustrySector>
        </Organisation>
        <Organisation>
            <!-- Valid record but not MAH so should be skipped if we filter properly -->
            <OrganisationId>ORG100000004</OrganisationId>
            <OrganisationName>Third MAH Inc</OrganisationName>
            <IndustrySector>marketing authorisation holder</IndustrySector>
        </Organisation>
    </Organisations>
    """

    # Create an in-memory zip file
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("spor_export.xml", xml_content)

    return output.getvalue()


@patch("requests.get")
def test_get_spor_organisations_resource_success(mock_get: Mock, mock_spor_xml_data: bytes) -> None:
    mock_response = Mock()
    mock_response.content = mock_spor_xml_data
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    resource = get_spor_organisations_resource("http://fake-spor-url.com")
    items = list(resource)

    # Expected items:
    # ORG100000000 (Valid MAH) -> Valid record
    # ORG100000001 (Not MAH) -> Skipped
    # ORG100000002 (Valid MAH) -> Valid record
    # ORG100000003 (Missing name) -> Skipped
    # ORG100000004 (Valid MAH lowercase) -> Valid record

    # We yield only MAH
    assert len(items) == 3

    assert items[0]["org_id"] == "ORG100000000"
    assert items[0]["org_name"] == "Valid MAH Pharma"

    assert items[1]["org_id"] == "ORG100000002"
    assert items[1]["org_name"] == "Second MAH Corp"

    assert items[2]["org_id"] == "ORG100000004"
    assert items[2]["org_name"] == "Third MAH Inc"


@patch("requests.get")
def test_get_spor_organisations_resource_validation_error(mock_get: Mock) -> None:
    # Minimal XML structure that mimics OMS SPOR export but with invalid data
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <Organisations>
        <Organisation>
            <!-- org id without matching text is filtered before validation -->
            <!-- Test validation error on empty string which is invalid per min_length=1 -->
            <OrganisationId>ORG100000000</OrganisationId>
            <OrganisationName>  </OrganisationName>
            <IndustrySector>Marketing Authorisation Holder</IndustrySector>
        </Organisation>
    </Organisations>
    """

    # Create an in-memory zip file
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("spor_export.xml", xml_content)

    mock_response = Mock()
    mock_response.content = output.getvalue()
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    resource = get_spor_organisations_resource("http://fake-spor-url.com")
    items = list(resource)

    assert len(items) == 0
