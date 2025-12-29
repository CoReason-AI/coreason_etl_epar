from datetime import datetime

import polars as pl
import pytest
from coreason_etl_epar.transform_silver import apply_scd2


@pytest.fixture  # type: ignore[misc]
def empty_history() -> pl.DataFrame:
    # Schema needs to match what we expect + overhead columns
    schema = {
        "id": pl.Int64,
        "data": pl.String,
        "valid_from": pl.Datetime,
        "valid_to": pl.Datetime,
        "is_current": pl.Boolean,
        "row_hash": pl.String,
    }
    return pl.DataFrame(schema=schema)


def test_scd2_initial_load(empty_history: pl.DataFrame) -> None:
    ts = datetime(2024, 1, 1)
    snapshot = pl.DataFrame({"id": [1, 2], "data": ["A", "B"]})

    result = apply_scd2(snapshot, empty_history, "id", ts, ["data"])

    assert result.height == 2
    assert result.filter(pl.col("id") == 1)["is_current"].item() is True
    assert result.filter(pl.col("id") == 1)["valid_from"].item() == ts
    assert result.filter(pl.col("id") == 1)["valid_to"].item() is None
    assert "row_hash" in result.columns


def test_scd2_no_change(empty_history: pl.DataFrame) -> None:
    ts1 = datetime(2024, 1, 1)
    ts2 = datetime(2024, 1, 2)

    # Setup history
    snapshot1 = pl.DataFrame({"id": [1], "data": ["A"]})
    history = apply_scd2(snapshot1, empty_history, "id", ts1, ["data"])

    # Same snapshot again
    result = apply_scd2(snapshot1, history, "id", ts2, ["data"])

    # Should be identical to input history (no new rows, no updates)
    assert result.height == 1
    assert result["valid_from"].item() == ts1
    assert result["valid_to"].item() is None
    assert result["is_current"].item() is True


def test_scd2_update(empty_history: pl.DataFrame) -> None:
    ts1 = datetime(2024, 1, 1)
    ts2 = datetime(2024, 1, 2)

    # Setup history
    snapshot1 = pl.DataFrame({"id": [1], "data": ["A"]})
    history = apply_scd2(snapshot1, empty_history, "id", ts1, ["data"])

    # Update data
    snapshot2 = pl.DataFrame({"id": [1], "data": ["A_Changed"]})
    result = apply_scd2(snapshot2, history, "id", ts2, ["data"])

    assert result.height == 2

    # Old record closed
    old_rec = result.filter((pl.col("id") == 1) & (pl.col("valid_from") == ts1))
    assert old_rec["is_current"].item() is False
    assert old_rec["valid_to"].item() == ts2

    # New record open
    new_rec = result.filter((pl.col("id") == 1) & (pl.col("valid_from") == ts2))
    assert new_rec["is_current"].item() is True
    assert new_rec["valid_to"].item() is None
    assert new_rec["data"].item() == "A_Changed"


def test_scd2_delete(empty_history: pl.DataFrame) -> None:
    ts1 = datetime(2024, 1, 1)
    ts2 = datetime(2024, 1, 2)

    # Setup history
    snapshot1 = pl.DataFrame({"id": [1], "data": ["A"]})
    history = apply_scd2(snapshot1, empty_history, "id", ts1, ["data"])

    # Empty snapshot (deleted)
    snapshot2 = pl.DataFrame(schema={"id": pl.Int64, "data": pl.String})
    result = apply_scd2(snapshot2, history, "id", ts2, ["data"])

    assert result.height == 1
    rec = result.row(0, named=True)
    assert rec["is_current"] is False
    assert rec["valid_to"] == ts2


def test_scd2_new_insert(empty_history: pl.DataFrame) -> None:
    ts1 = datetime(2024, 1, 1)
    ts2 = datetime(2024, 1, 2)

    # Setup history
    snapshot1 = pl.DataFrame({"id": [1], "data": ["A"]})
    history = apply_scd2(snapshot1, empty_history, "id", ts1, ["data"])

    # Add new id
    snapshot2 = pl.DataFrame({"id": [1, 2], "data": ["A", "B"]})
    result = apply_scd2(snapshot2, history, "id", ts2, ["data"])

    # Expect: 1 unchanged, 2 new
    assert result.height == 2

    rec1 = result.filter(pl.col("id") == 1).row(0, named=True)
    assert rec1["is_current"] is True
    assert rec1["valid_from"] == ts1  # Original TS

    rec2 = result.filter(pl.col("id") == 2).row(0, named=True)
    assert rec2["is_current"] is True
    assert rec2["valid_from"] == ts2


def test_scd2_resurrection() -> None:
    # History: Record ID=1 is CLOSED (valid_to is set, is_current=False)
    # Snapshot: Record ID=1 reappears.
    # Expected: New active record inserted. Old one stays closed.

    schema = {
        "id": pl.Int64,
        "data": pl.String,
        "valid_from": pl.Datetime,
        "valid_to": pl.Datetime,
        "is_current": pl.Boolean,
        "row_hash": pl.String,
    }

    ts_old = datetime(2023, 1, 1)
    ts_close = datetime(2023, 2, 1)
    ts_new = datetime(2024, 1, 1)

    history = pl.DataFrame(
        {
            "id": [1],
            "data": ["A"],
            "valid_from": [ts_old],
            "valid_to": [ts_close],
            "is_current": [False],
            "row_hash": ["hash_A"],
        },
        schema=schema,
    )

    snapshot = pl.DataFrame({"id": [1], "data": ["A"]})

    result = apply_scd2(snapshot, history, "id", ts_new, ["data"])

    # Check results: Should have 2 rows.
    # 1. Old closed row.
    # 2. New open row.

    assert result.height == 2

    # Old row check
    old_row = result.filter(pl.col("valid_from") == ts_old)
    assert old_row["is_current"].item() is False
    assert old_row["valid_to"].item() == ts_close

    # New row check
    new_row = result.filter(pl.col("valid_from") == ts_new)
    assert new_row["is_current"].item() is True
    assert new_row["valid_to"].item() is None


def test_scd2_flapping() -> None:
    # History: A -> B (Active is B).
    # Snapshot: A (Reverting to old value).
    # Expected: Close B, Insert A (new version).

    schema = {
        "id": pl.Int64,
        "data": pl.String,
        "valid_from": pl.Datetime,
        "valid_to": pl.Datetime,
        "is_current": pl.Boolean,
        "row_hash": pl.String,
    }

    ts1 = datetime(2024, 1, 1)
    ts2 = datetime(2024, 1, 2)
    ts3 = datetime(2024, 1, 3)

    history = pl.DataFrame(
        {
            "id": [1, 1],
            "data": ["A", "B"],
            "valid_from": [ts1, ts2],
            "valid_to": [ts2, None],
            "is_current": [False, True],
            "row_hash": ["hash_A", "hash_B"],  # Mock hashes
        },
        schema=schema,
    )

    snapshot = pl.DataFrame({"id": [1], "data": ["A"]})  # Back to A

    result = apply_scd2(snapshot, history, "id", ts3, ["data"])

    assert result.height == 3

    # Verify B is closed
    row_b = result.filter(pl.col("valid_from") == ts2)
    assert row_b["is_current"].item() is False
    assert row_b["valid_to"].item() == ts3

    # Verify new A is open
    row_a_new = result.filter(pl.col("valid_from") == ts3)
    assert row_a_new["is_current"].item() is True
    assert row_a_new["data"].item() == "A"
