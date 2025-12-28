import polars as pl
from datetime import datetime
import hashlib
from typing import List, Optional

def generate_row_hash(df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
    """
    Generates an MD5 hash of the specified columns for each row.
    Adds a 'row_hash' column.
    """
    # Concatenate columns as string and hash
    # We must ensure consistent ordering and formatting

    # Select columns, cast to string, fill nulls with empty string to ensure stability
    # Then concat and hash

    # Note: polars doesn't have a direct row_hash function, we construct it.

    expr = pl.concat_str(
        [pl.col(c).cast(pl.String).fill_null("") for c in columns],
        separator="|"
    )

    # Use map_elements to apply md5 (slow but standard) or if polars has a hash function?
    # Polars has `hash()` but it's not MD5 (it's 64-bit non-cryptographic).
    # FRD specifies MD5.
    # We can use `map_elements` with hashlib.

    return df.with_columns(
        row_hash = expr.map_elements(lambda x: hashlib.md5(x.encode()).hexdigest(), return_dtype=pl.String)
    )

def apply_scd2(
    current_snapshot: pl.DataFrame,
    history: pl.DataFrame,
    primary_key: str,
    ingestion_ts: datetime,
    hash_columns: List[str]
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
    # Add row_hash to snapshot
    snapshot_hashed = generate_row_hash(current_snapshot, hash_columns)

    # 2. Identify Changes
    # Join Snapshot with Current History (is_current=True)

    if history.is_empty():
        # Initial Load: All are new
        return snapshot_hashed.with_columns(
            valid_from = pl.lit(ingestion_ts),
            valid_to = pl.lit(None, dtype=pl.Datetime),
            is_current = pl.lit(True)
        )

    current_history = history.filter(pl.col("is_current") == True)
    closed_history = history.filter(pl.col("is_current") == False)

    # Join on PK
    # Left join snapshot -> history to find New + Changed + Unchanged
    # Anti join history -> snapshot to find Deleted

    # We need to distinguish:
    # - New: PK in snapshot, not in history
    # - Changed: PK in both, hash differs
    # - Unchanged: PK in both, hash matches
    # - Deleted: PK in history, not in snapshot

    # Rename history columns to avoid collision
    hist_renamed = current_history.select([primary_key, "row_hash"]).rename({"row_hash": "hist_row_hash"})

    joined = snapshot_hashed.join(hist_renamed, on=primary_key, how="left")

    # New Records
    new_records = joined.filter(pl.col("hist_row_hash").is_null()).drop("hist_row_hash")

    # Changed Records (New Version)
    changed_records = joined.filter(
        (pl.col("hist_row_hash").is_not_null()) &
        (pl.col("row_hash") != pl.col("hist_row_hash"))
    ).drop("hist_row_hash")

    # Unchanged Records (We don't touch these in terms of creating new rows,
    # but we need to keep the existing history rows for them)
    # Actually, we just need to identify the keys to NOT close in history.

    # 3. Process Updates

    # Keys to Close:
    # 1. Deleted records (in current_history but not in snapshot)
    # 2. Changed records (in current_history AND in snapshot AND hash diff)

    # Get keys of changed records
    changed_keys = changed_records.select(primary_key)

    # Get keys of deleted records
    # specific logic: present in current_history but not in snapshot
    deleted_keys = current_history.select(primary_key).join(
        snapshot_hashed.select(primary_key), on=primary_key, how="anti"
    )

    keys_to_close = pl.concat([changed_keys, deleted_keys]).unique()

    # Update History: Close records
    # We construct a boolean mask or join to update 'valid_to' and 'is_current'

    # Since polars LazyFrame update is tricky, we can reconstruct the history.

    # Updated Old Records:
    # Filter history for keys_to_close -> set valid_to = ingestion_ts, is_current = False
    # Filter history for NOT keys_to_close -> keep as is

    # But wait, 'history' contains closed records too. We only close 'current' records.
    # So we split 'current_history' into 'to_close' and 'to_keep'.

    history_to_close = current_history.join(keys_to_close, on=primary_key, how="inner")
    history_to_keep = current_history.join(keys_to_close, on=primary_key, how="anti")

    closed_updates = history_to_close.with_columns(
        valid_to = pl.lit(ingestion_ts),
        is_current = pl.lit(False)
    )

    # 4. Create New Entries
    # From New Records and Changed Records

    new_entries = pl.concat([new_records, changed_records]).with_columns(
        valid_from = pl.lit(ingestion_ts),
        valid_to = pl.lit(None, dtype=pl.Datetime),
        is_current = pl.lit(True)
    )

    # 5. Union All
    # Result = closed_history + history_to_keep + closed_updates + new_entries

    final_history = pl.concat([
        closed_history,
        history_to_keep,
        closed_updates,
        new_entries
    ], how="diagonal") # diagonal to handle potential column reordering or missing cols if schema evolved (though strict schema preferred)

    return final_history
