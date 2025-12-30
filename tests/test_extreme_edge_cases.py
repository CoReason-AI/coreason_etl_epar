from datetime import datetime
from typing import Any, Dict

import polars as pl
import pytest
from coreason_etl_epar.transform_silver import apply_scd2
from coreason_etl_epar.transform_enrich import enrich_epar

def test_scd2_schema_drift_resilience() -> None:
    """
    Edge Case: Source adds a new column 'extra_col' that is NOT in history.
    SCD2 should ignore it and output history schema.
    """
    ts = datetime(2024, 1, 1)
    schema: Dict[str, pl.DataType | Any] = {
        "id": pl.Int64,
        "data": pl.String,
        "valid_from": pl.Datetime,
        "valid_to": pl.Datetime,
        "is_current": pl.Boolean,
        "row_hash": pl.String,
    }
    history = pl.DataFrame(schema=schema)

    # Snapshot has extra column
    snapshot = pl.DataFrame({
        "id": [1],
        "data": ["V1"],
        "extra_col": ["Noise"]
    })

    result = apply_scd2(snapshot, history, "id", ts, ["data"])

    assert "extra_col" not in result.columns
    assert result.schema["data"] == pl.String

def test_scd2_null_in_hash_cols() -> None:
    """
    Edge Case: Hashing columns contain Nulls.
    Should be treated as empty string or distinct from "Null" string, depending on logic.
    Current logic: `pl.col(c).cast(pl.String).fill_null("")`.
    So Null and "" are indistinguishable.
    """
    ts = datetime(2024, 1, 1)
    schema: Dict[str, pl.DataType | Any] = {
        "id": pl.Int64,
        "data": pl.String,
        "valid_from": pl.Datetime,
        "valid_to": pl.Datetime,
        "is_current": pl.Boolean,
        "row_hash": pl.String,
    }
    history = pl.DataFrame(schema=schema)

    # Snapshot 1: Null
    s1 = pl.DataFrame({"id": [1], "data": [None]})
    h1 = apply_scd2(s1, history, "id", ts, ["data"])

    # Snapshot 2: Empty String
    s2 = pl.DataFrame({"id": [1], "data": [""]})
    h2 = apply_scd2(s2, h1, "id", ts, ["data"])

    # Since fill_null("") is used, Null -> ""
    # So h1 row hash should equal s2 row hash.
    # Therefore, NO update should occur in h2.

    assert h2.height == 1 # Still 1 row, no update
    assert h2["is_current"].item() is True

def test_enrich_unicode_nightmare() -> None:
    """
    Edge Case: Input strings with Zero Width Spaces, Control Characters, Emojis.
    """
    dirty_mah = "Big Pharma\u200b Inc." # Zero width space
    clean_mah = "Big Pharma Inc."

    df = pl.DataFrame({
        "product_number": ["P1"],
        "medicine_name": ["M"],
        "active_substance": ["S"],
        "atc_code": ["A"],
        "marketing_authorisation_holder": [dirty_mah],
        "authorisation_status": ["A"],
    })

    spor_df = pl.DataFrame({
        "name": [clean_mah],
        "org_id": ["ORG-001"]
    })

    result = enrich_epar(df, spor_df)

    # Collect result to check
    mah_id = result["spor_mah_id"].to_list()[0]
    assert mah_id == "ORG-001"

def test_gold_creation_empty_silver() -> None:
    """
    Edge Case: Running Gold transformation on completely empty Silver data.
    Should return empty DataFrames with CORRECT schema, not fail.
    """
    from coreason_etl_epar.transform_gold import create_gold_layer

    # Empty Silver with correct schema
    silver_schema = {
        "product_number": pl.String,
        "base_procedure_id": pl.String, # Added
        "medicine_name": pl.String,
        "active_substance_list": pl.List(pl.String),
        "atc_code_list": pl.List(pl.String),
        "therapeutic_area": pl.String,
        "valid_from": pl.Datetime,
        "valid_to": pl.Datetime,
        "is_current": pl.Boolean,
        "status_normalized": pl.String,
        "spor_mah_id": pl.String,
        "biosimilar": pl.Boolean,
        "generic": pl.Boolean,
        "orphan": pl.Boolean,
        "url": pl.String,
    }
    silver = pl.DataFrame(schema=silver_schema)

    gold = create_gold_layer(silver)

    assert "dim_medicine" in gold
    assert "fact_regulatory_history" in gold
    assert gold["dim_medicine"].height == 0
    # Check column existence to ensure schema is preserved
    assert "coreason_id" in gold["dim_medicine"].columns
