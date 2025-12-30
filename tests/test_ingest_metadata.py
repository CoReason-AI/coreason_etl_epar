import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from coreason_etl_epar.ingest import epar_index, spor_organisations


@pytest.fixture
def dummy_excel_file(tmp_path: Path) -> str:
    file_path = tmp_path / "medicines.xlsx"
    data = {
        "Category": ["Human"],
        "Product number": ["EMEA/H/C/001234"],
        "Medicine name": ["Med A"],
        "Marketing authorisation holder": ["Holder A"],
        "Active substance": ["Sub A"],
        "Authorisation status": ["Authorised"],
        "URL": ["http://a.com"],
    }
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)
    return str(file_path)


@pytest.fixture
def dummy_spor_zip(tmp_path: Path) -> str:
    zip_path = tmp_path / "spor.zip"
    root = ET.Element("Organisations")
    org = ET.SubElement(root, "Organisation")
    ET.SubElement(org, "OrganisationId").text = "ORG-1001"
    ET.SubElement(org, "Name").text = "Pharma Corp"
    roles = ET.SubElement(org, "Roles")
    role = ET.SubElement(roles, "Role")
    ET.SubElement(role, "Name").text = "Marketing Authorisation Holder"

    tree = ET.ElementTree(root)
    xml_path = tmp_path / "export.xml"
    tree.write(xml_path)

    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(xml_path, arcname="export.xml")
    return str(zip_path)


def test_epar_bronze_metadata(dummy_excel_file: str) -> None:
    resource = epar_index(dummy_excel_file)
    rows = list(resource)
    assert len(rows) == 1
    row = rows[0]

    # Assert Metadata presence
    assert "source_file_hash" in row, "Missing source_file_hash in EPAR row"
    assert "ingestion_ts" in row, "Missing ingestion_ts in EPAR row"
    assert "raw_payload" in row, "Missing raw_payload in EPAR row"

    # Assert Metadata content validity
    assert isinstance(row["source_file_hash"], str)
    assert len(row["source_file_hash"]) > 0
    assert isinstance(row["ingestion_ts"], str)
    assert isinstance(row["raw_payload"], dict)
    assert row["raw_payload"]["product_number"] == "EMEA/H/C/001234"


def test_spor_bronze_metadata(dummy_spor_zip: str) -> None:
    resource = spor_organisations(dummy_spor_zip)
    rows = list(resource)
    assert len(rows) == 1
    row = rows[0]

    # Assert Metadata presence
    assert "source_file_hash" in row, "Missing source_file_hash in SPOR row"
    assert "ingestion_ts" in row, "Missing ingestion_ts in SPOR row"
    assert "raw_payload" in row, "Missing raw_payload in SPOR row"

    assert isinstance(row["source_file_hash"], str)
    assert len(row["source_file_hash"]) > 0
    assert isinstance(row["ingestion_ts"], str)
    assert isinstance(row["raw_payload"], dict)
    # Check that raw payload has the extracted data
    assert row["raw_payload"]["name"] == "Pharma Corp"
