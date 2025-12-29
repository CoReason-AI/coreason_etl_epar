from datetime import datetime
from typing import Any, Dict

import polars as pl

from coreason_etl_epar.transform_silver import apply_scd2
from coreason_etl_epar.transform_enrich import enrich_epar

def test_scd2_conflicting_duplicates() -> None:
    """
    Edge Case: Source snapshot contains duplicate primary keys with conflicting data.
    The pipeline must be deterministic (pick first) and not fail or create duplicate history entries for the same active period.
    """
    ts = datetime(2024, 1, 1)

    # Schema for history
    schema: Dict[str, pl.DataType | Any] = {
        "id": pl.Int64,
        "data": pl.String,
        "valid_from": pl.Datetime,
        "valid_to": pl.Datetime,
        "is_current": pl.Boolean,
        "row_hash": pl.String,
    }

    empty_history = pl.DataFrame(schema=schema)

    # Snapshot with duplicates
    # Row 1: ID=1, Data="Winner"
    # Row 2: ID=1, Data="Loser"
    snapshot = pl.DataFrame({
        "id": [1, 1],
        "data": ["Winner", "Loser"]
    })

    result = apply_scd2(snapshot, empty_history, "id", ts, ["data"])

    # Check that we have exactly 1 active record for ID=1
    active = result.filter(pl.col("is_current"))
    assert active.height == 1

    # Check that it picked the first one (Polars unique(keep='first') behavior)
    val = active["data"].item()
    assert val == "Winner"


def test_enrich_chaos_strings() -> None:
    """
    Edge Case: Very messy strings in columns that are exploded.
    """
    # messy_substance: " A  /  B +  + / C " -> Should become ["A", "B", "C"]
    # messy_atc: "A01,, B02 ; ; C03" -> Should become ["A01", "B02", "C03"] (normalized)

    df = pl.DataFrame({
        "product_number": ["P1"],
        "medicine_name": ["M"],
        "active_substance": [" A  /  B +  + / C "],
        "atc_code": ["A01,, B02 ; ; C03"],
        "marketing_authorisation_holder": ["MAH"],
        "authorisation_status": ["A"]
    })

    spor_df = pl.DataFrame(schema={"name": pl.String, "org_id": pl.String})

    result = enrich_epar(df, spor_df)

    # Check Substance
    subs = result["active_substance_list"].to_list()[0]
    assert sorted(subs) == ["A", "B", "C"]
    assert "" not in subs

    # Check ATC
    # Note: Enrich normalizes ATC. "A01" is invalid format (too short), "B02" invalid.
    # The valid regex is ^[A-Z]\d{2}[A-Z]{2}\d{2}$ (e.g. A01BC01).
    # "A01", "B02", "C03" do NOT match the regex, so they should be filtered OUT.
    # Wait, let's verify regex behavior.
    # A01 is L7? No, it's L3.
    # So actually, if I input short codes, the result list should be EMPTY.

    atc = result["atc_code_list"].to_list()[0]
    assert atc == []  # All filtered out because they don't match L7 format
