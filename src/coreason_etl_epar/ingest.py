# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_epar

import datetime
import io
from collections.abc import Generator
from typing import Any, cast

import dlt
import pandas as pd
import requests  # type: ignore[import-untyped]
from pydantic import ValidationError

from coreason_etl_epar.schemas import EPARSourceRow
from coreason_etl_epar.utils.logger import logger


@dlt.resource(name="epar_index")  # type: ignore[misc]
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
        df = df[df["category"] == "Human"]
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
            if isinstance(v, (pd.Timestamp, datetime.datetime)):
                row_dict[k] = v.isoformat()
            else:
                row_dict[k] = v

        # Source business flags often use "Yes"/"No"
        for flag in ["generic", "biosimilar", "orphan", "conditional_approval", "exceptional_circumstances"]:
            if flag in row_dict and isinstance(row_dict[flag], str):
                val = row_dict[flag].strip().lower()
                if val == "yes" or val == "true":
                    row_dict[flag] = True
                elif val == "no" or val == "false":
                    row_dict[flag] = False
                else:
                    row_dict[flag] = False

        if "category" not in row_dict:
            row_dict["category"] = "Human"

        try:
            # Validate row
            valid_row = EPARSourceRow.model_validate(row_dict)
            yield valid_row.model_dump()
        except ValidationError as e:
            # Route to quarantine
            error_details = {"raw_data": row_dict_raw, "error": str(e)}
            yield cast("dict[str, Any]", dlt.mark.with_table_name(error_details, "epar_index_quarantine"))
