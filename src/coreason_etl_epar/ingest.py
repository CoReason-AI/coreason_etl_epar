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
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Generator
from typing import Any, cast

import dlt
import pandas as pd
import requests
from pydantic import ValidationError

from coreason_etl_epar.schemas import EPARSourceRow, SPOROrganisationRow
from coreason_etl_epar.utils.logger import logger


@dlt.resource(name="epar_index")
def get_epar_index_resource(url: str) -> Generator[dict[str, Any]]:
    """
    Ingests the EPAR Excel index.
    Downloads the file, explicitly filters out 'Veterinary' records,
    validates rows, and routes failures to quarantine.
    """
    logger.info(f"Downloading EPAR index from {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    logger.info("Reading Excel content into DataFrame")
    # Read excel file from content
    df = pd.read_excel(io.BytesIO(response.content), skiprows=8)

    # Clean column names to align with schema
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("\n", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )

    column_mapping = {
        "international_non-proprietary_name_inn___common_name": "active_substance",
        "active_substance": "active_substance",
        "orphan_medicine": "orphan",
        "marketing_authorisation_holder_company_name": "marketing_authorisation_holder",
    }
    df.rename(columns=column_mapping, inplace=True)

    # Filter out Veterinary
    if "category" in df.columns:
        logger.info("Filtering out 'Veterinary' records")
        original_count = len(df)
        df = df[df["category"] == "Human"]
        veterinary_drop_count = original_count - len(df)
        logger.info(
            "Veterinary records filtered",
            veterinary_drop_count=veterinary_drop_count,
        )
    else:
        logger.warning("Category column not found, assuming all Human")

    logger.info(f"Processing {len(df)} rows")
    # Drop rows where essential columns are completely missing
    if "product_number" in df.columns:
        df = df.dropna(subset=["product_number"])

    for _, row in df.iterrows():
        # Clean row data
        row_dict: dict[str, Any] = {}
        row_dict_raw: dict[str, Any] = {str(k): v for k, v in row.to_dict().items()}
        for k, v in row_dict_raw.items():
            if pd.isna(v):
                continue
            row_dict[k] = v

        if "category" not in row_dict:
            row_dict["category"] = "Human"

        try:
            # Validate row via Pydantic (natively coerces "Yes"/"No" strings and pd.Timestamp)
            valid_row = EPARSourceRow.model_validate(row_dict)
            yield valid_row.model_dump()
        except ValidationError as e:
            # Route to quarantine
            error_details = {"raw_data": row_dict_raw, "error": str(e)}
            yield cast("dict[str, Any]", dlt.mark.with_table_name(error_details, "epar_index_quarantine"))


@dlt.resource(name="spor_organisations_master")
def get_spor_organisations_resource(url: str) -> Generator[dict[str, Any]]:
    """
    Ingests the SPOR OMS Bulk Export.
    Downloads the zip file, streams the XML, and extracts organizations.
    Filters for "Marketing Authorisation Holder" role if possible.
    """
    logger.info(f"Downloading SPOR OMS export from {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    # The API returns a ZIP file containing an XML file
    logger.info("Extracting XML from ZIP in memory")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Assuming there is only one XML file in the zip
        xml_filename = next(name for name in z.namelist() if name.endswith(".xml"))

        with z.open(xml_filename) as xml_file:
            logger.info(f"Streaming XML content from {xml_filename}")

            # Use iterparse to stream the XML without loading the entire document into memory
            # nosec B314: This is parsing trusted data from a reliable European API.
            context = ET.iterparse(xml_file, events=("end",))  # noqa: S314

            # Define the namespace map for easier querying, assuming standard OMS SPOR namespace
            # If there's a specific namespace, we might need to adjust, but let's strip namespaces or use local names
            for _event, elem in context:
                # Assuming each organization is represented by an 'Organisation' or similar element
                # We'll strip namespaces for simplicity if present
                local_name = elem.tag.split("}", 1)[-1]

                if local_name == "Organisation":
                    org_id = None
                    org_name = None
                    is_mah = False

                    for child in elem.iter():
                        child_local_name = child.tag.split("}", 1)[-1]
                        if child_local_name == "OrganisationId" and org_id is None:
                            org_id = child.text
                        elif child_local_name == "OrganisationName" and org_name is None:
                            org_name = child.text

                        if child.text and "marketing authorisation holder" in child.text.strip().lower():
                            is_mah = True

                    if org_id and org_name and is_mah:
                        # The requirement says "If possible, limit to roles
                        # 'Marketing Authorisation Holder' to reduce volume"
                        # We filter to MAH if we found the string anywhere in the
                        # organization element.
                        try:
                            valid_row = SPOROrganisationRow(org_id=org_id.strip(), org_name=org_name.strip())
                            yield valid_row.model_dump()
                        except ValidationError as e:
                            logger.warning(f"Skipping invalid organization record {org_id}: {e}")

                    # Clear the element to save memory
                    elem.clear()
