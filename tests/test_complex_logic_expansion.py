from datetime import datetime
from typing import Any, Dict

import polars as pl
import pytest
from polars.exceptions import ColumnNotFoundError

from coreason_etl_epar.transform_silver import apply_scd2


def test_scd2_schema_evolution_ignore_new_cols() -> None:
    """
    Test that new columns in the source snapshot are ignored if they are not part of the
    target schema or hash_cols. The pipeline should not crash, but the new data won't be persisted
    unless the history schema is migrated (which is out of scope for this function).
    """
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
    ts = datetime(2024, 1, 1)

    # Source has an extra column "new_col"
    snapshot = pl.DataFrame({"id": [1], "data": ["V1"], "new_col": ["Ignored"]})

    # apply_scd2 typically selects relevant columns or joins.
    # If it strictly selects based on 'cols' argument, "new_col" is ignored.
    # If it joins everything, it might fail if history lacks the column.

    # Let's see behavior. We expect it to succeed but NOT have "new_col" in output.
    result = apply_scd2(snapshot, history, pk, ts, ["data"])

    assert "new_col" not in result.columns
    assert result.height == 1
    assert result["data"].item() == "V1"


def test_scd2_missing_hash_col_in_source() -> None:
    """
    Test failure when a required business column (used for hashing) is missing from source.
    """
    schema: Dict[str, pl.DataType | Any] = {
        "id": pl.Int64,
        "data": pl.String,
        "valid_from": pl.Datetime,
        "valid_to": pl.Datetime,
        "is_current": pl.Boolean,
        "row_hash": pl.String,
    }
    history = pl.DataFrame(schema=schema)

    # Source misses "data" column
    snapshot = pl.DataFrame({"id": [1], "other": ["V1"]})

    # Should raise ColumnNotFoundError
    with pytest.raises(ColumnNotFoundError):
        apply_scd2(snapshot, history, "id", datetime(2024, 1, 1), ["data"])


def test_scd2_bootstrap_empty_history() -> None:
    """
    Test apply_scd2 with a completely empty history (no schema).
    This covers the 'bootstrap' branch where schema is inferred from snapshot.
    """
    history = pl.DataFrame()  # No schema
    snapshot = pl.DataFrame({"id": [1], "data": ["V1"]})
    ts = datetime(2024, 1, 1)

    result = apply_scd2(snapshot, history, "id", ts, ["data"])

    # Result should have columns from snapshot + SCD meta
    assert "data" in result.columns
    assert "valid_from" in result.columns
    assert result.height == 1
