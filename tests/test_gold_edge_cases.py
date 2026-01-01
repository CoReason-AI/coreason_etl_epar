# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_epar

import polars as pl

from coreason_etl_epar.transform_gold import create_gold_layer, generate_coreason_id


def test_gold_therapeutic_area_dirty_splitting() -> None:
    """
    Verify that therapeutic_area splitting handles:
    - Empty strings
    - Consecutive delimiters (";;")
    - Leading/Trailing whitespace
    """

    # Setup minimal Silver DF
    # Note: Logic moved to Silver (clean_epar_bronze), so Silver DF must have pre-cleaned list.
    # This test now verifies that Gold correctly ingests the list.
    # We simulate the result of the Silver cleaning here.
    data = {
        "product_number": ["P1", "P2"],  # Required for ID generation
        "coreason_id": ["C1", "C2"],
        "medicine_name": ["M1", "M2"],
        "base_procedure_id": ["P1", "P2"],
        "is_current": [True, True],
        "brand_name": ["M1", "M2"],
        "is_biosimilar": [False, False],
        "is_generic": [False, False],
        "is_orphan": [False, False],
        "ema_product_url": ["u1", "u2"],
        # Pre-Cleaned Lists
        "therapeutic_area_list": [
            ["Cardiology", "Oncology"],  # Clean
            ["Neurology", "Dermatology"],  # Result of cleaning "  Neurology ; ; Dermatology; "
        ],
        # Required columns for other logic (even if empty lists)
        "atc_code_list": [[], []],
        "active_substance_list": [[], []],
        "status_normalized": ["Approved", "Approved"],
        "valid_from": [None, None],
        "valid_to": [None, None],
        "spor_mah_id": [None, None],
        "url": ["u1", "u2"],
        "biosimilar": [False, False],
        "generic": [False, False],
        "orphan": [False, False],
    }

    # Note: schema might be inferred, but lists need care.
    silver_df = pl.DataFrame(
        data,
        schema_overrides={
            "atc_code_list": pl.List(pl.String),
            "active_substance_list": pl.List(pl.String),
            "therapeutic_area_list": pl.List(pl.String),
        },
    )

    gold = create_gold_layer(silver_df)

    bridge = gold["bridge_medicine_features"]

    # Filter for Therapeutic Area
    ta_bridge = bridge.filter(pl.col("feature_type") == "THERAPEUTIC_AREA")

    # Check P1 (Clean)
    id_p1 = generate_coreason_id("P1")
    c1_vals = ta_bridge.filter(pl.col("coreason_id") == id_p1)["feature_value"].to_list()
    assert set(c1_vals) == {"Cardiology", "Oncology"}

    # Check P2 (Dirty)
    id_p2 = generate_coreason_id("P2")
    c2_vals = ta_bridge.filter(pl.col("coreason_id") == id_p2)["feature_value"].to_list()

    # Expected: "Neurology", "Dermatology"
    # Should NOT contain: "", " ", etc.
    assert "Neurology" in c2_vals
    assert "Dermatology" in c2_vals
    assert "" not in c2_vals
    assert " " not in c2_vals
    assert len(c2_vals) == 2
