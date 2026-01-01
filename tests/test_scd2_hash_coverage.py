# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_epar

from datetime import datetime
from typing import Any, Dict

import polars as pl
import pytest

from coreason_etl_epar.transform_silver import apply_scd2, generate_row_hash

# The list of columns we EXPECT to be covered.
# This must match (or be a subset of) the logic in pipeline.py
EXPECTED_HASH_COLS = [
    "authorisation_status",
    "medicine_name",
    "marketing_authorisation_holder",
    "active_substance",
    "atc_code",
    "therapeutic_area_list",
    "generic",
    "biosimilar",
    "orphan",
    "conditional_approval",
    "exceptional_circumstances",
    "url",
]


@pytest.fixture
def base_history() -> pl.DataFrame:
    schema: Dict[str, Any] = {
        "product_number": pl.String,
        "authorisation_status": pl.String,
        "medicine_name": pl.String,
        "marketing_authorisation_holder": pl.String,
        "active_substance": pl.String,
        "atc_code": pl.String,
        "therapeutic_area": pl.String,
        "therapeutic_area_list": pl.List(pl.String),
        "generic": pl.Boolean,
        "biosimilar": pl.Boolean,
        "orphan": pl.Boolean,
        "conditional_approval": pl.Boolean,
        "exceptional_circumstances": pl.Boolean,
        "url": pl.String,
        # SCD Cols
        "valid_from": pl.Datetime,
        "valid_to": pl.Datetime,
        "is_current": pl.Boolean,
        "row_hash": pl.String,
    }

    # Create one initial record
    data = {
        "product_number": ["P1"],
        "authorisation_status": ["Authorised"],
        "medicine_name": ["MedA"],
        "marketing_authorisation_holder": ["HolderA"],
        "active_substance": ["SubA"],
        "atc_code": ["A01"],
        "therapeutic_area": ["AreaA"],
        "therapeutic_area_list": [["AreaA"]],
        "generic": [False],
        "biosimilar": [False],
        "orphan": [False],
        "conditional_approval": [False],
        "exceptional_circumstances": [False],
        "url": ["http://ema.europa.eu/p1"],
        "valid_from": [datetime(2023, 1, 1)],
        "valid_to": [None],
        "is_current": [True],
        "row_hash": ["placeholder"],
    }

    df = pl.DataFrame(data, schema=schema)

    # Calculate REAL hash so no-change detection works
    # Note: We must drop existing row_hash first or overwrite it
    df = df.drop("row_hash")
    df = generate_row_hash(df, EXPECTED_HASH_COLS)

    # generate_row_hash adds "row_hash" column
    return df


def test_scd2_detects_business_column_changes(base_history: pl.DataFrame) -> None:
    """
    Iterates over each business column, changes its value in a new snapshot,
    and verifies that apply_scd2 (using the FULL list of columns) detects the change.
    """
    ts_update = datetime(2024, 1, 1)

    # Base snapshot data (matching history)
    base_snapshot_data = {
        "product_number": "P1",
        "authorisation_status": "Authorised",
        "medicine_name": "MedA",
        "marketing_authorisation_holder": "HolderA",
        "active_substance": "SubA",
        "atc_code": "A01",
        "therapeutic_area": "AreaA",
        "therapeutic_area_list": ["AreaA"],
        "generic": False,
        "biosimilar": False,
        "orphan": False,
        "conditional_approval": False,
        "exceptional_circumstances": False,
        "url": "http://ema.europa.eu/p1",
    }

    for col in EXPECTED_HASH_COLS:
        # Create a modified snapshot
        modified_data = base_snapshot_data.copy()

        # Change value based on type
        current_val = modified_data[col]
        if isinstance(current_val, bool):
            modified_data[col] = not current_val
        elif isinstance(current_val, str):
            modified_data[col] = current_val + "_CHANGED"
        elif isinstance(current_val, list):
            modified_data[col] = current_val + ["NEW_ITEM"]

        snapshot = pl.DataFrame([modified_data])

        # Run SCD2 with the EXPECTED list
        # We assume the pipeline uses this list. This test verifies logic IF the list is used.
        # The integration is implicitly tested because we updated pipeline.py to use this list.
        result = apply_scd2(snapshot, base_history, "product_number", ts_update, EXPECTED_HASH_COLS)

        # Assert we have 2 rows (old closed, new open)
        assert result.height == 2, f"Failed to detect change in column: {col}"

        new_rec = result.filter(pl.col("valid_from") == ts_update)
        assert not new_rec.is_empty()
        assert new_rec["is_current"].item() is True

        # Verify value change in result
        expected_val = modified_data[col]
        # Use to_list()[0] to safely get Python object (handling List columns)
        actual_val = new_rec[col].to_list()[0]

        # If actual_val is a Series (Polars List vs Python List ambiguity), convert
        if isinstance(actual_val, pl.Series):
            actual_val = actual_val.to_list()

        assert actual_val == expected_val


def test_scd2_ignores_irrelevant_changes(base_history: pl.DataFrame) -> None:
    """
    Verify that changing a column NOT in the hash list does NOT trigger update.
    """
    ts_update = datetime(2024, 1, 1)

    base_snapshot_data = {
        "product_number": "P1",
        "authorisation_status": "Authorised",
        "medicine_name": "MedA",
        "marketing_authorisation_holder": "HolderA",
        "active_substance": "SubA",
        "atc_code": "A01",
        "therapeutic_area": "AreaA",
        "generic": False,
        "biosimilar": False,
        "orphan": False,
        "conditional_approval": False,
        "exceptional_circumstances": False,
        "url": "http://ema.europa.eu/p1",
        # Extra column not in hash list
        "revision_date": datetime(2024, 1, 1),  # Changed from history (implicitly None or old)
        "therapeutic_area_list": ["AreaA"],  # Must be present to match history schema
    }

    snapshot = pl.DataFrame([base_snapshot_data])

    # Run SCD2 using the standard list
    result = apply_scd2(snapshot, base_history, "product_number", ts_update, EXPECTED_HASH_COLS)

    # Should be NO update (still 1 row, valid_from unchanged)
    assert result.height == 1
    assert result["valid_from"].item() == datetime(2023, 1, 1)
