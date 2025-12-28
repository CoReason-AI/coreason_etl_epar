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
