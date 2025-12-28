import hashlib
from datetime import datetime
from typing import List

import polars as pl


def generate_row_hash(df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
    """
    Generates an MD5 hash of the specified columns for each row.
    Adds a 'row_hash' column.
    """
    expr = pl.concat_str([pl.col(c).cast(pl.String).fill_null("") for c in columns], separator="|")

    return df.with_columns(
        row_hash=expr.map_elements(lambda x: hashlib.md5(x.encode()).hexdigest(), return_dtype=pl.String)
    )


def apply_scd2(
    current_snapshot: pl.DataFrame,
    history: pl.DataFrame,
    primary_key: str,
    ingestion_ts: datetime,
    hash_columns: List[str],
) -> pl.DataFrame:
    """
    Applies SCD Type 2 logic to merge a new snapshot into the existing history.

    Args:
        current_snapshot: The new data (Bronze)
        history: The existing history (Silver). Schema must include:
                 [primary_key, ..., valid_from, valid_to, is_current, row_hash]
        primary_key: The column name for the join key.
        ingestion_ts: The timestamp for valid_from/valid_to.
        hash_columns: Columns used to detect changes.

    Returns:
        Updated history DataFrame.
    """

    # 1. Prepare Snapshot
    snapshot_hashed = generate_row_hash(current_snapshot, hash_columns)

    # 2. Identify Changes
    if history.is_empty():
        return snapshot_hashed.with_columns(
            valid_from=pl.lit(ingestion_ts), valid_to=pl.lit(None, dtype=pl.Datetime), is_current=pl.lit(True)
        )

    current_history = history.filter(pl.col("is_current"))
    closed_history = history.filter(~pl.col("is_current"))

    # Rename history columns to avoid collision
    hist_renamed = current_history.select([primary_key, "row_hash"]).rename({"row_hash": "hist_row_hash"})

    joined = snapshot_hashed.join(hist_renamed, on=primary_key, how="left")

    # New Records
    new_records = joined.filter(pl.col("hist_row_hash").is_null()).drop("hist_row_hash")

    # Changed Records (New Version)
    changed_records = joined.filter(
        (pl.col("hist_row_hash").is_not_null()) & (pl.col("row_hash") != pl.col("hist_row_hash"))
    ).drop("hist_row_hash")

    # 3. Process Updates
    changed_keys = changed_records.select(primary_key)
    deleted_keys = current_history.select(primary_key).join(
        snapshot_hashed.select(primary_key), on=primary_key, how="anti"
    )

    keys_to_close = pl.concat([changed_keys, deleted_keys]).unique()

    history_to_close = current_history.join(keys_to_close, on=primary_key, how="inner")
    history_to_keep = current_history.join(keys_to_close, on=primary_key, how="anti")

    closed_updates = history_to_close.with_columns(valid_to=pl.lit(ingestion_ts), is_current=pl.lit(False))

    # 4. Create New Entries
    new_entries = pl.concat([new_records, changed_records]).with_columns(
        valid_from=pl.lit(ingestion_ts), valid_to=pl.lit(None, dtype=pl.Datetime), is_current=pl.lit(True)
    )

    # 5. Union All
    final_history = pl.concat([closed_history, history_to_keep, closed_updates, new_entries], how="diagonal")

    return final_history
