from datetime import datetime
from typing import Any, Dict

import polars as pl
import pytest
from polars.exceptions import InvalidOperationError, SchemaError

from coreason_etl_epar.transform_silver import apply_scd2


def test_scd2_duplicate_source_rows() -> None:
    """
    Complex Case: Source snapshot contains duplicate Primary Keys.
    The logic must deduplicate before processing to avoid exploding history or crashing.
    Expectation: The function uses .unique(keep="first") so only one version persists.
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

    # Snapshot with duplicates: ID 1 appears twice with DIFFERENT data.
    # Logic should pick first (arbitrary but deterministic based on input order/sort)
    snapshot = pl.DataFrame({"id": [1, 1], "data": ["V1-A", "V1-B"]})

    result = apply_scd2(snapshot, history, "id", ts, ["data"])

    # Assert Deduplication
    assert result.height == 1
    row = result.row(0, named=True)
    assert row["id"] == 1
    assert row["data"] == "V1-A"  # First one kept


def test_scd2_source_type_mutation() -> None:
    """
    Complex Case: Source column type changes from History.
    History 'data' is String. Snapshot provides 'data' as Int.

    If the snapshot has Int64 and history has String,
    polars might raise SchemaError when trying to join or stack them if they are not explicitly cast first.

    The apply_scd2 logic:
    1. snapshot_unique = current_snapshot.unique(...) -> Int64 column 'data'.
    2. generate_row_hash -> casts to String for hashing. OK.
    3. join -> joined. 'data' is still Int64 in snapshot part.
    4. concat([closed_history, history_to_keep... new_entries]).
       Polars `concat` requires schemas to match or supertype.
       Int64 vs String -> Incompatible.

    The current implementation does NOT cast snapshot cols to history schema BEFORE processing/concat.
    It does it at the very end: `return final_history.select([pl.col(c).cast(history.schema[c])...])`

    BUT `pl.concat` happens BEFORE that return.
    So this SHOULD fail with SchemaError.

    This verifies that our pipeline is STRICT and will alert us (via crash) on schema drift.
    """
    ts = datetime(2024, 1, 1)
    # Use strict typing for schema to satisfy mypy
    # Instantiate types (pl.Int64()) to match pl.DataType
    schema: Dict[str, pl.DataType] = {
        "id": pl.Int64(),
        "data": pl.String(),
        "valid_from": pl.Datetime(),
        "valid_to": pl.Datetime(),
        "is_current": pl.Boolean(),
        "row_hash": pl.String(),
    }
    # Explicitly cast to strict schema
    history = pl.DataFrame(
        {"id": [1], "data": ["Old"], "valid_from": [ts], "valid_to": [None], "is_current": [True]}
    ).with_columns(pl.lit("hash_old").alias("row_hash"))

    # We use cast with dict
    history = history.cast(schema)  # type: ignore[arg-type]

    # Snapshot provides Int!
    ts2 = datetime(2024, 1, 2)
    snapshot = pl.DataFrame({"id": [1], "data": [123]})

    # Expect SchemaError during concatenation
    with pytest.raises(SchemaError):
        apply_scd2(snapshot, history, "id", ts2, ["data"])


def test_scd2_source_type_mutation_incompatible() -> None:
    """
    Complex Case: Source column incompatible type.
    History 'count' is Int. Snapshot provides 'count' as "NotAnInt" string.

    Even if we fixed the SchemaError above (by pre-casting),
    casting "NotAnInt" to Int64 would raise InvalidOperationError (strict).
    """
    ts = datetime(2024, 1, 1)
    schema: Dict[str, pl.DataType | Any] = {
        "id": pl.Int64,
        "count": pl.Int64,
        "valid_from": pl.Datetime,
        "valid_to": pl.Datetime,
        "is_current": pl.Boolean,
        "row_hash": pl.String,
    }
    history = pl.DataFrame(schema=schema)

    # Snapshot provides non-numeric string
    snapshot = pl.DataFrame({"id": [1], "count": ["NotAnInt"]})

    # This might fail with SchemaError first (String vs Int64 in concat).
    # If we bypassed that, it would fail with InvalidOperationError on cast.
    # We catch either to verify it fails safely (doesn't silently corrupt).

    with pytest.raises((SchemaError, InvalidOperationError)):
        apply_scd2(snapshot, history, "id", ts, ["count"])
