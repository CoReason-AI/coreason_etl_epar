from datetime import datetime
from typing import Any, Dict

import polars as pl

from coreason_etl_epar.transform_silver import apply_scd2, clean_epar_bronze


def test_scd2_order_invariance() -> None:
    """
    Test Complex Case: Reordering of multi-value fields should NOT trigger SCD2 updates.
    Day 1: "A; B"
    Day 2: "B; A"
    Result: Day 2 should produce exact same hash as Day 1 -> No History Update.
    """
    ts = datetime(2024, 1, 1)

    # 1. Setup Bronze Data (Simulating Source)
    # Day 1
    bronze_d1 = pl.DataFrame(
        {
            "product_number": ["P1"],
            "therapeutic_area": ["Cancer; Diabetes"],
            "active_substance": ["Sub A + Sub B"],
            "category": ["Human"],
            # Mock other required cols for cleaning
            "medicine_name": ["M"],
            "marketing_authorisation_holder": ["MAH"],
            "authorisation_status": ["Authorised"],
            "url": ["u"],
            "atc_code": ["A01"],
            "generic": [False],
            "biosimilar": [False],
            "orphan": [False],
            "conditional_approval": [False],
            "exceptional_circumstances": [False],
        }
    )

    # Day 2: Reordered
    bronze_d2 = pl.DataFrame(
        {
            "product_number": ["P1"],
            "therapeutic_area": ["Diabetes; Cancer"],  # Changed order
            "active_substance": ["Sub B + Sub A"],  # Changed order
            "category": ["Human"],
            "medicine_name": ["M"],
            "marketing_authorisation_holder": ["MAH"],
            "authorisation_status": ["Authorised"],
            "url": ["u"],
            "atc_code": ["A01"],
            "generic": [False],
            "biosimilar": [False],
            "orphan": [False],
            "conditional_approval": [False],
            "exceptional_circumstances": [False],
        }
    )

    # 2. Clean Data (Where sorting happens)
    silver_d1 = clean_epar_bronze(bronze_d1)
    silver_d2 = clean_epar_bronze(bronze_d2)

    # Verify normalization worked
    list_d1 = silver_d1["therapeutic_area_list"][0].to_list()
    list_d2 = silver_d2["therapeutic_area_list"][0].to_list()

    assert list_d1 == ["Cancer", "Diabetes"]  # Sorted
    assert list_d2 == ["Cancer", "Diabetes"]  # Sorted, identical to d1

    # 3. Apply SCD2 Day 1
    # Schema setup for history
    schema: Dict[str, pl.DataType | Any] = {
        "product_number": pl.String,
        "therapeutic_area_list": pl.List(pl.String),
        "active_substance_list": pl.List(pl.String),
        "atc_code_list": pl.List(pl.String),
        # Other cols
        "medicine_name": pl.String,
        "marketing_authorisation_holder": pl.String,
        "status_normalized": pl.String,
        "url": pl.String,
        "base_procedure_id": pl.String,
        "generic": pl.Boolean,
        "biosimilar": pl.Boolean,
        "orphan": pl.Boolean,
        "conditional_approval": pl.Boolean,
        "exceptional_circumstances": pl.Boolean,
        # SCD
        "valid_from": pl.Datetime,
        "valid_to": pl.Datetime,
        "is_current": pl.Boolean,
        "row_hash": pl.String,
    }
    history = pl.DataFrame(schema=schema)

    # Define Hash Cols (Must match pipeline)
    hash_cols = [
        "therapeutic_area_list",
        "active_substance_list",
        # ... others
    ]

    history_d1 = apply_scd2(silver_d1, history, "product_number", ts, hash_cols)
    assert history_d1.height == 1
    row_hash_d1 = history_d1["row_hash"][0]

    # 4. Apply SCD2 Day 2
    ts_d2 = datetime(2024, 1, 2)
    history_d2 = apply_scd2(silver_d2, history_d1, "product_number", ts_d2, hash_cols)

    # 5. Assert No Change
    # Should still be 1 row, valid_to is Null
    assert history_d2.height == 1
    assert history_d2["valid_to"][0] is None
    assert history_d2["row_hash"][0] == row_hash_d1

    # Verify logic: if sorting was NOT applied, row_hash would differ and we'd get a new row.


def test_mixed_delimiters_robustness() -> None:
    """
    Test Complex Case: Mixed delimiters in Therapeutic Area.
    "A; B, C" -> ["A", "B", "C"]
    """
    df = pl.DataFrame(
        {
            "product_number": ["P1"],
            "therapeutic_area": ["Area A; Area B, Area C"],
            "medicine_name": ["M"],
            "category": ["Human"],
        }
    )

    cleaned = clean_epar_bronze(df)

    res = cleaned["therapeutic_area_list"][0].to_list()
    assert res == ["Area A", "Area B", "Area C"]
