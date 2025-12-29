import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest
from dlt.extract.exceptions import ResourceExtractionError

from coreason_etl_epar.ingest import spor_organisations


@pytest.fixture
def dummy_spor_zip(tmp_path: Path) -> str:
    zip_path = tmp_path / "spor.zip"

    # Create XML content
    # Structure mimics what the ingestion expects
    root = ET.Element("Organisations")

    # Org 1: Valid MAH
    org1 = ET.SubElement(root, "Organisation")
    ET.SubElement(org1, "OrganisationId").text = "ORG-1001"
    ET.SubElement(org1, "Name").text = "Pharma Corp A"
    roles1 = ET.SubElement(org1, "Roles")
    role1a = ET.SubElement(roles1, "Role")
    ET.SubElement(role1a, "Name").text = "Marketing Authorisation Holder"

    # Org 2: Invalid Role
    org2 = ET.SubElement(root, "Organisation")
    ET.SubElement(org2, "OrganisationId").text = "ORG-1002"
    ET.SubElement(org2, "Name").text = "Logistics Co"
    roles2 = ET.SubElement(org2, "Roles")
    role2a = ET.SubElement(roles2, "Role")
    ET.SubElement(role2a, "Name").text = "Distributor"

    # Org 3: Valid MAH (mixed case/structure check)
    org3 = ET.SubElement(root, "Organisation")
    ET.SubElement(org3, "OrganisationId").text = "ORG-1003"
    ET.SubElement(org3, "Name").text = "BioTech Inc"
    roles3 = ET.SubElement(org3, "Roles")
    role3a = ET.SubElement(roles3, "Role")
    ET.SubElement(role3a, "Name").text = "Marketing authorisation holder"  # Case insensitive check

    # Org 4: Valid MAH (Role text directly in Role node, not Name child - edge case support)
    org4 = ET.SubElement(root, "Organisation")
    ET.SubElement(org4, "OrganisationId").text = "ORG-1004"
    ET.SubElement(org4, "Name").text = "Direct Role Corp"
    roles4 = ET.SubElement(org4, "Roles")
    role4a = ET.SubElement(roles4, "Role")
    role4a.text = "Marketing Authorisation Holder"

    tree = ET.ElementTree(root)
    xml_path = tmp_path / "export.xml"
    tree.write(xml_path)

    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(xml_path, arcname="export.xml")

    return str(zip_path)


def test_spor_organisations_resource(dummy_spor_zip: str) -> None:
    resource = spor_organisations(dummy_spor_zip)
    rows = list(resource)

    # Expected: Org 1, Org 3, Org 4 (Total 3)
    # Org 2 is Distributor

    assert len(rows) == 3

    ids = [r["org_id"] for r in rows]
    assert "ORG-1001" in ids
    assert "ORG-1003" in ids
    assert "ORG-1004" in ids
    assert "ORG-1002" not in ids


def test_spor_no_xml(tmp_path: Path) -> None:
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("readme.txt", "nothing here")

    resource = spor_organisations(str(zip_path))
    rows = list(resource)
    assert len(rows) == 0


def test_spor_bad_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad.zip"
    with open(zip_path, "wb") as f:
        f.write(b"not a zip")

    # Check that dlt raises ResourceExtractionError which wraps the BadZipFile
    with pytest.raises(ResourceExtractionError) as excinfo:
        list(spor_organisations(str(zip_path)))
    assert "File is not a zip file" in str(excinfo.value)


def test_spor_general_exception(tmp_path: Path) -> None:
    # This test is harder to trigger because zipfile handles opening.
    # We can mock zipfile.ZipFile to raise a generic exception
    from unittest.mock import patch

    with patch("zipfile.ZipFile", side_effect=Exception("Generic Error")):
        with pytest.raises(ResourceExtractionError) as excinfo:
            list(spor_organisations("dummy_path"))
        assert "Generic Error" in str(excinfo.value)
