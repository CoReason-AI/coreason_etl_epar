import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from typing import Any, Dict, Iterator, List

import dlt
import polars as pl
from loguru import logger
from pydantic import ValidationError

from coreason_etl_epar.schema import EPARSourceRow


@dlt.resource(name="epar_index", write_disposition="replace")
def epar_index(file_path: str) -> Iterator[Dict[str, Any] | Any]:
    """
    DLT Resource to ingest EPAR Index from Excel.
    Filters for Category == 'Human' and excludes 'Veterinary'.
    """
    logger.info(f"Reading EPAR index from {file_path}")

    # Read Excel using Polars
    try:
        df = pl.read_excel(file_path)
    except Exception as e:
        logger.error(f"Failed to read Excel file: {e}")
        raise

    # Clean column names (normalize to snake_case for Pydantic mapping)
    df = df.rename({col: col.strip().lower().replace(" ", "_") for col in df.columns})

    # Filter for 'Human' category
    if "category" not in df.columns:
        logger.error("Column 'category' not found in Excel file")
        return

    total_rows = df.height
    filtered_df = df.filter(pl.col("category").str.strip_chars().str.to_uppercase() == "HUMAN")
    human_rows = filtered_df.height
    veterinary_drop_count = total_rows - human_rows

    logger.bind(veterinary_drop_count=veterinary_drop_count, metric="veterinary_drop_count").info(
        f"Filtered {veterinary_drop_count} Veterinary/Other rows"
    )

    # Normalize 'category' to 'Human' to satisfy Pydantic Strict Literal
    filtered_df = filtered_df.with_columns(pl.lit("Human").alias("category"))

    for row in filtered_df.iter_rows(named=True):
        try:
            validated_row = EPARSourceRow(**row)
            yield validated_row.model_dump()

        except ValidationError as e:
            # Handle None or Missing product_number
            product_number = row.get("product_number")
            if not product_number:
                product_number = "UNKNOWN"

            logger.warning(f"Validation failed for row {product_number}: {e}")

            quarantine_record = {
                "raw_data": row,
                "error_message": str(e),
                "product_number": product_number,
                "ingestion_ts": datetime.now().isoformat(),
            }
            yield dlt.mark.with_table_name(quarantine_record, "_quarantine")


@dlt.resource(name="spor_organisations", write_disposition="replace")
def spor_organisations(file_path: str) -> Iterator[Dict[str, Any]]:
    """
    DLT Resource to ingest SPOR Organisations from a Zip containing XML.
    Extracts organizations and filters for 'Marketing Authorisation Holder' role.
    """
    logger.info(f"Reading SPOR organisations from {file_path}")

    try:
        with zipfile.ZipFile(file_path, "r") as z:
            # Assume there is one XML file or we pick the first one ending in .xml
            xml_files = [f for f in z.namelist() if f.endswith(".xml")]
            if not xml_files:
                logger.error("No XML file found in SPOR zip archive")
                return

            with z.open(xml_files[0]) as f:
                # Use iterparse for memory efficiency on large XML
                context = ET.iterparse(f, events=("end",))

                for _event, elem in context:
                    if elem.tag.endswith("rganisation"):  # Handle potential namespace prefixes
                        org_data: Dict[str, Any] = {}
                        roles: List[str] = []

                        # Extract children data
                        for child in elem:
                            tag_name = child.tag.split("}")[-1]  # Strip namespace
                            if tag_name.lower() == "name":
                                org_data["name"] = child.text
                            elif tag_name.lower() == "organisationid":
                                org_data["org_id"] = child.text
                            elif tag_name.lower() == "roles":
                                # Iterate over roles
                                for role in child:
                                    # Role text might be in a text node or a child 'Name' node
                                    role_name = role.text
                                    if not role_name:
                                        # Try finding a name child
                                        for role_child in role:
                                            if role_child.tag.split("}")[-1].lower() == "name":
                                                role_name = role_child.text
                                                break
                                    if role_name:
                                        roles.append(role_name)

                        org_data["roles"] = roles

                        # Filter for 'Marketing Authorisation Holder'
                        # Per FRD: "Filter: If possible, limit to roles 'Marketing Authorisation Holder'"
                        is_mah = any("marketing authorisation holder" in r.lower() for r in roles)

                        if is_mah:
                            yield {"org_id": org_data.get("org_id"), "name": org_data.get("name"), "roles": roles}

                        # Clear element to save memory
                        elem.clear()

    except zipfile.BadZipFile:
        logger.error(f"Invalid zip file: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to process SPOR file: {e}")
        raise
