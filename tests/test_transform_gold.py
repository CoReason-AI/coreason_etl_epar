from datetime import datetime

import polars as pl

from coreason_etl_epar.transform_gold import create_gold_layer, generate_coreason_id


def test_generate_coreason_id() -> None:
    id1 = generate_coreason_id("EMEA/H/C/001")
    id2 = generate_coreason_id("EMEA/H/C/001")
    id3 = generate_coreason_id("EMEA/H/C/002")

    assert id1 == id2
    assert id1 != id3


def test_create_gold_layer() -> None:
    # Setup Enriched Silver Data
    silver_df = pl.DataFrame(
        {
            "product_number": ["EMEA/H/C/001", "EMEA/H/C/001"],
            "medicine_name": ["Med A", "Med A"],
            "base_procedure_id": ["001", "001"],
            "biosimilar": [False, False],
            "generic": [False, False],
            "orphan": [True, True],
            "url": ["http://a", "http://a"],
            "status_normalized": ["CONDITIONAL_APPROVAL", "APPROVED"],
            "valid_from": [datetime(2024, 1, 1), datetime(2024, 2, 1)],
            "valid_to": [datetime(2024, 2, 1), None],
            "is_current": [False, True],
            "spor_mah_id": ["ORG-1", "ORG-1"],
            "active_substance_list": [["Sub A"], ["Sub A"]],
            "atc_code_list": [["A01"], ["A01"]],
            "therapeutic_area": ["Cancer", "Cancer"],
        }
    )

    gold = create_gold_layer(silver_df)

    # Verify Dim Medicine
    dim = gold["dim_medicine"]
    assert dim.height == 1  # Deduplicated
    row = dim.row(0, named=True)
    assert row["medicine_name"] == "Med A"
    assert row["is_orphan"] is True
    assert "coreason_id" in row

    # Verify Fact History
    fact = gold["fact_regulatory_history"]
    assert fact.height == 2  # Full history
    assert fact.filter(pl.col("status") == "APPROVED")["is_current"].item() is True
    assert fact.filter(pl.col("status") == "CONDITIONAL_APPROVAL")["valid_to"].is_not_null().item() is True

    # Verify Bridge
    bridge = gold["bridge_medicine_features"]
    # 1 Substance + 1 ATC + 1 Area = 3 rows
    assert bridge.height == 3

    feat_types = bridge["feature_type"].to_list()
    assert "SUBSTANCE" in feat_types
    assert "ATC_CODE" in feat_types
    assert "THERAPEUTIC_AREA" in feat_types

    assert bridge.filter(pl.col("feature_type") == "SUBSTANCE")["feature_value"].item() == "Sub A"


def test_gold_layer_therapeutic_split() -> None:
    silver_df = pl.DataFrame(
        {
            "product_number": ["P1"],
            "medicine_name": ["M1"],
            "base_procedure_id": ["1"],
            "biosimilar": [False],
            "generic": [False],
            "orphan": [False],
            "url": ["u"],
            "status_normalized": ["APPROVED"],
            "valid_from": [datetime(2024, 1, 1)],
            "valid_to": [None],
            "is_current": [True],
            "spor_mah_id": ["O1"],
            "active_substance_list": [["S1"]],
            "atc_code_list": [["A1"]],
            "therapeutic_area": ["Area 1; Area 2"],  # Split check
        }
    )

    gold = create_gold_layer(silver_df)
    bridge = gold["bridge_medicine_features"]

    area_rows = bridge.filter(pl.col("feature_type") == "THERAPEUTIC_AREA")
    assert area_rows.height == 2
    vals = area_rows["feature_value"].to_list()
    assert "Area 1" in vals
    assert "Area 2" in vals


def test_gold_fallback_no_current() -> None:
    # Test defensive line:
    # current_df = df.sort("valid_from", descending=True).unique(subset=["coreason_id"], keep="first")
    # This checks the fallback logic when no record is marked as current

    silver_df = pl.DataFrame(
        {
            "product_number": ["P1", "P1"],
            "medicine_name": ["M1_old", "M1_new"],
            "base_procedure_id": ["1", "1"],
            "biosimilar": [False, False],
            "generic": [False, False],
            "orphan": [False, False],
            "url": ["u", "u"],
            "status_normalized": ["APPROVED", "WITHDRAWN"],
            "valid_from": [datetime(2024, 1, 1), datetime(2024, 2, 1)],
            "valid_to": [datetime(2024, 2, 1), datetime(2024, 3, 1)],  # Both closed
            "is_current": [False, False],
            "spor_mah_id": ["O1", "O1"],
            "active_substance_list": [["S1"], ["S1"]],
            "atc_code_list": [["A1"], ["A1"]],
            "therapeutic_area": ["A", "A"],
        }
    )

    gold = create_gold_layer(silver_df)
    dim = gold["dim_medicine"]
    assert dim.height == 1
    # Should pick the one with latest valid_from (M1_new)
    assert dim["medicine_name"].item() == "M1_new"


def test_gold_empty_lists() -> None:
    # Silver data has empty lists for substance/ATC.
    # Gold bridge table should not fail, just have fewer rows.

    # Define schema explicitly to avoid Null type inference
    schema = {
        "product_number": pl.String,
        "medicine_name": pl.String,
        "base_procedure_id": pl.String,
        "biosimilar": pl.Boolean,
        "generic": pl.Boolean,
        "orphan": pl.Boolean,
        "url": pl.String,
        "status_normalized": pl.String,
        "valid_from": pl.Datetime,
        "valid_to": pl.Datetime,
        "is_current": pl.Boolean,
        "spor_mah_id": pl.String,
        "active_substance_list": pl.List(pl.String),
        "atc_code_list": pl.List(pl.String),
        "therapeutic_area": pl.String,
    }

    silver_df = pl.DataFrame(
        {
            "product_number": ["P1"],
            "medicine_name": ["M1"],
            "base_procedure_id": ["1"],
            "biosimilar": [False],
            "generic": [False],
            "orphan": [False],
            "url": ["u"],
            "status_normalized": ["A"],
            "valid_from": [datetime(2024, 1, 1)],
            "valid_to": [None],
            "is_current": [True],
            "spor_mah_id": ["O1"],
            "active_substance_list": [[]],  # Empty
            "atc_code_list": [[]],  # Empty
            "therapeutic_area": [None],  # Null
        },
        schema=schema,
    )

    gold = create_gold_layer(silver_df)
    bridge = gold["bridge_medicine_features"]

    # Should be empty because all inputs are empty/null
    assert bridge.height == 0
