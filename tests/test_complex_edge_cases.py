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

from coreason_etl_epar.transform_enrich import jaro_winkler
from coreason_etl_epar.transform_gold import create_gold_layer
from coreason_etl_epar.transform_silver import apply_scd2


def test_scd2_same_timestamp_update() -> None:
    """
    Edge Case: An update arrives with the SAME timestamp as the current record's valid_from.
    This can happen if the pipeline runs twice quickly or clock skew.

    Expected Behavior:
    The logic currently uses `valid_from` to filter current.
    If `valid_from` == `ingestion_ts`, we might close the record with `valid_to` = `ingestion_ts`.
    This results in `valid_from` == `valid_to`, effectively a zero-duration record.
    The new record is inserted with `valid_from` = `ingestion_ts`.

    We want to ensure we don't crash or create overlapping active records.
    """
    ts = datetime(2024, 1, 1)

    schema = {
        "id": pl.Int64,
        "data": pl.String,
        "valid_from": pl.Datetime,
        "valid_to": pl.Datetime,
        "is_current": pl.Boolean,
        "row_hash": pl.String,
    }

    # Existing history: Created at TS
    history = pl.DataFrame(
        {
            "id": [1],
            "data": ["A"],
            "valid_from": [ts],
            "valid_to": [None],
            "is_current": [True],
            "row_hash": ["hash_A"],
        },
        schema=schema,
    )

    # New snapshot: Changed data, SAME TS
    snapshot = pl.DataFrame({"id": [1], "data": ["B"]})

    # We rely on internal hashing, but here we just need to ensure the logic flows.
    # Note: apply_scd2 calculates hashes internally.
    # We must ensure the snapshot generates a DIFFERENT hash for this to be an update.
    # "B" != "A", so it will be an update.

    result = apply_scd2(snapshot, history, "id", ts, ["data"])

    # Check what happened to the old record
    old_rec = result.filter(pl.col("data") == "A")
    # It should be closed at TS
    assert old_rec["valid_to"].item() == ts
    assert old_rec["is_current"].item() is False

    # Check new record
    new_rec = result.filter(pl.col("data") == "B")
    assert new_rec["valid_from"].item() == ts
    assert new_rec["valid_to"].item() is None
    assert new_rec["is_current"].item() is True

    # Conclusion: The old record effectively becomes zero-duration (valid_from == valid_to).
    # This is acceptable for SCD Type 2.


def test_gold_empty_schema() -> None:
    """
    Edge Case: Silver input is completely empty (e.g. first run, no matches).
    Gold output must still have the correct schema (columns), not just be empty structureless DFs.
    """
    # Empty Silver DF with correct schema
    silver_schema: Dict[str, pl.DataType | Any] = {
        "product_number": pl.String,
        "medicine_name": pl.String,
        "marketing_authorisation_holder": pl.String,
        "active_substance": pl.String,
        "authorisation_status": pl.String,
        "url": pl.String,
        "revision_date": pl.Datetime,
        # Enriched cols
        "base_procedure_id": pl.String,
        "active_substance_list": pl.List(pl.String),
        "atc_code_list": pl.List(pl.String),
        "status_normalized": pl.String,
        "spor_mah_id": pl.String,
        "therapeutic_area": pl.String,
        "biosimilar": pl.Boolean,
        "generic": pl.Boolean,
        "orphan": pl.Boolean,
        # SCD cols
        "valid_from": pl.Datetime,
        "valid_to": pl.Datetime,
        "is_current": pl.Boolean,
    }

    empty_silver = pl.DataFrame(schema=silver_schema)

    gold = create_gold_layer(empty_silver)

    # Check dim_medicine
    dim = gold["dim_medicine"]
    assert dim.height == 0
    assert "coreason_id" in dim.columns
    assert "medicine_name" in dim.columns
    assert "brand_name" in dim.columns

    # Check fact_regulatory_history
    fact = gold["fact_regulatory_history"]
    assert fact.height == 0
    assert "history_id" in fact.columns
    assert "status" in fact.columns

    # Check bridge
    bridge = gold["bridge_medicine_features"]
    assert bridge.height == 0
    assert "feature_type" in bridge.columns
    assert "feature_value" in bridge.columns


def test_jaro_winkler_edge_cases() -> None:
    """
    Test Jaro-Winkler implementation against edge cases.
    """
    # 1. Identity
    assert jaro_winkler("abc", "abc") == 1.0

    # 2. Complete mismatch
    assert jaro_winkler("abc", "xyz") == 0.0

    # 3. Empty strings
    assert jaro_winkler("", "") == 1.0
    assert jaro_winkler("a", "") == 0.0
    assert jaro_winkler("", "a") == 0.0

    # 4. Unicode
    # 'café' vs 'cafe'
    # Distance should be high but not 1.0
    score_unicode = jaro_winkler("café", "cafe")
    assert 0.8 < score_unicode < 1.0

    # 5. Case sensitivity (Implementation is case-sensitive, caller usually lowercases)
    # Check raw function behavior
    assert jaro_winkler("A", "a") == 0.0

    # 6. Prefix boost (Winkler)
    # "Martha" vs "Marhta" (Match first 3) -> High score
    # "Dwayne" vs "Duane" (Match first 1) -> Lower score
    s1 = jaro_winkler("Martha", "Marhta")
    s2 = jaro_winkler("Dwayne", "Duane")
    assert s1 > s2

    # 7. Long strings (Performance/correctness check)
    long_a = "a" * 100
    long_b = "a" * 100
    assert jaro_winkler(long_a, long_b) == 1.0
