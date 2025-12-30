from datetime import datetime
from typing import Any, Dict

import polars as pl

from coreason_etl_epar.transform_silver import apply_scd2


def test_scd2_full_lifecycle() -> None:
    """
    Integration Test: Verifies the full lifecycle of a single entity through SCD Type 2 changes.
    Lifecycle: Insert -> Update -> Delete -> Resurrect -> Update.
    """

    # Schema setup
    schema: Dict[str, pl.DataType | Any] = {
        "id": pl.Int64,
        "data": pl.String,
        "valid_from": pl.Datetime,
        "valid_to": pl.Datetime,
        "is_current": pl.Boolean,
        "row_hash": pl.String,
    }

    history = pl.DataFrame(schema=schema)
    pk = "id"
    cols = ["data"]

    # Day 1: Insert (Data="V1")
    ts1 = datetime(2024, 1, 1)
    snap1 = pl.DataFrame({"id": [1], "data": ["V1"]})
    history = apply_scd2(snap1, history, pk, ts1, cols)

    # Assert Day 1
    assert history.height == 1
    row1 = history.row(0, named=True)
    assert row1["data"] == "V1"
    assert row1["valid_from"] == ts1
    assert row1["valid_to"] is None
    assert row1["is_current"] is True

    # Day 2: Update (Data="V2")
    ts2 = datetime(2024, 1, 2)
    snap2 = pl.DataFrame({"id": [1], "data": ["V2"]})
    history = apply_scd2(snap2, history, pk, ts2, cols)

    # Assert Day 2
    # Should have 2 rows: V1 closed, V2 open
    assert history.height == 2

    v1_row = history.filter(pl.col("valid_from") == ts1).row(0, named=True)
    assert v1_row["is_current"] is False
    assert v1_row["valid_to"] == ts2

    v2_row = history.filter(pl.col("valid_from") == ts2).row(0, named=True)
    assert v2_row["is_current"] is True
    assert v2_row["valid_to"] is None
    assert v2_row["data"] == "V2"

    # Day 3: Delete (Record missing from snapshot)
    ts3 = datetime(2024, 1, 3)
    snap3 = pl.DataFrame(schema={"id": pl.Int64, "data": pl.String})  # Empty
    history = apply_scd2(snap3, history, pk, ts3, cols)

    # Assert Day 3
    # V2 should be closed. No new row.
    assert history.height == 2  # Still 2 rows, just updated

    v2_row = history.filter(pl.col("valid_from") == ts2).row(0, named=True)
    assert v2_row["is_current"] is False
    assert v2_row["valid_to"] == ts3

    # Day 4: Resurrection (Data="V2" - same as before, or "V3". Let's say "V2" again ("Restored"))
    ts4 = datetime(2024, 1, 4)
    snap4 = pl.DataFrame({"id": [1], "data": ["V2"]})
    history = apply_scd2(snap4, history, pk, ts4, cols)

    # Assert Day 4
    # Should have 3 rows. New open row for V2 (starting ts4).
    # Note: Even though data is "V2" (same as prev), it's a new validity period because of the gap.
    assert history.height == 3

    v2_restored = history.filter(pl.col("valid_from") == ts4).row(0, named=True)
    assert v2_restored["is_current"] is True
    assert v2_restored["valid_to"] is None
    assert v2_restored["data"] == "V2"

    # Day 5: Update (Data="V3")
    ts5 = datetime(2024, 1, 5)
    snap5 = pl.DataFrame({"id": [1], "data": ["V3"]})
    history = apply_scd2(snap5, history, pk, ts5, cols)

    # Assert Day 5
    # Should have 4 rows. V2 (restored) closed. V3 open.
    assert history.height == 4

    v2_restored = history.filter(pl.col("valid_from") == ts4).row(0, named=True)
    assert v2_restored["is_current"] is False
    assert v2_restored["valid_to"] == ts5

    v3_row = history.filter(pl.col("valid_from") == ts5).row(0, named=True)
    assert v3_row["is_current"] is True
    assert v3_row["data"] == "V3"


def test_scd2_multiple_updates_same_day() -> None:
    """
    Edge Case: Updates happening with same timestamp (or re-running same day).
    If ran twice same day with SAME data -> Idempotent.
    If ran twice same day with DIFFERENT data -> Should update correctly (handle intra-day change?).

    Note: Standard SCD2 using 'valid_from' as PK part often assumes daily granularity.
    If we run multiple times, the 'valid_to' of prev and 'valid_from' of new will collide.
    The logic in `apply_scd2` sets `valid_to = ingestion_ts` and new `valid_from = ingestion_ts`.
    If ingestion_ts is identical, we get `valid_from == valid_to`, which is a zero-duration record.
    This is generally acceptable (it existed for 0 time) or it effectively replaces it.
    Let's verify behavior.
    """
    schema = {
        "id": pl.Int64,
        "data": pl.String,
        "valid_from": pl.Datetime,
        "valid_to": pl.Datetime,
        "is_current": pl.Boolean,
        "row_hash": pl.String,
    }
    history = pl.DataFrame(schema=schema)
    pk = "id"
    cols = ["data"]
    ts = datetime(2024, 1, 1)

    # Run 1: V1
    snap1 = pl.DataFrame({"id": [1], "data": ["V1"]})
    history = apply_scd2(snap1, history, pk, ts, cols)

    # Run 2: V2 (Same Timestamp!) - e.g. Correction
    snap2 = pl.DataFrame({"id": [1], "data": ["V2"]})
    history = apply_scd2(snap2, history, pk, ts, cols)

    # Expectation:
    # Row 1 (V1): valid_from=ts, valid_to=ts, is_current=False
    # Row 2 (V2): valid_from=ts, valid_to=None, is_current=True

    assert history.height == 2

    v1 = history.filter(pl.col("data") == "V1").row(0, named=True)
    assert v1["valid_from"] == v1["valid_to"]  # Zero duration
    assert v1["is_current"] is False

    v2 = history.filter(pl.col("data") == "V2").row(0, named=True)
    assert v2["valid_from"] == ts
    assert v2["is_current"] is True
