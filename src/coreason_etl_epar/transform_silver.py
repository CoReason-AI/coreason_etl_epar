import hashlib
from datetime import datetime
from typing import List

import polars as pl


def get_status_normalization_expr(col_name: str) -> pl.Expr:
    """
    Returns a Polars expression to normalize Authorisation Status.
    Strictly prioritized: Refused > Withdrawn > Suspended > Conditional > Exceptional > Approved.
    """
    col = pl.col(col_name).str.strip_chars().str.to_uppercase()
    return (
        pl.when(col.str.contains("REFUSED"))
        .then(pl.lit("REJECTED"))
        .when(col.str.contains("WITHDRAWN"))
        .then(pl.lit("WITHDRAWN"))
        .when(col.str.contains("SUSPENDED"))
        .then(pl.lit("SUSPENDED"))
        .when(col.str.contains("CONDITIONAL"))
        .then(pl.lit("CONDITIONAL_APPROVAL"))
        .when(col.str.contains("EXCEPTIONAL"))
        .then(pl.lit("EXCEPTIONAL_CIRCUMSTANCES"))
        # FIX: Ensure "AUTHORISED" does not match "NOT AUTHORISED"
        # We require it contains "AUTHORISED" AND DOES NOT contain "NOT "
        # We assume "NOT " (with space) or just "NOT" is enough signal for negation.
        .when(col.str.contains("AUTHORISED") & ~col.str.contains("NOT "))
        .then(pl.lit("APPROVED"))
        .otherwise(pl.lit("UNKNOWN"))
    )


def clean_epar_bronze(df: pl.DataFrame) -> pl.DataFrame:
    """
    Cleans and normalizes the Bronze EPAR DataFrame.
    - Strips invisible characters (\\u200b).
    - Normalizes multi-value fields (substance, atc_code).
    - Normalizes status.
    - Derives base_procedure_id.
    """
    # Use lazy API for efficiency, then collect
    lf = df.lazy()

    # 1. Strip Invisible Characters from ALL String columns
    # We target specific columns known to be strings or select by type
    # For safety, we target the business columns.
    string_cols = [
        "medicine_name",
        "marketing_authorisation_holder",
        "active_substance",
        "atc_code",
        "therapeutic_area",
        "authorisation_status",
        "url",
        "product_number",
    ]
    # Check if cols exist (some might be missing in older versions, but schema enforces them)
    existing_cols = [c for c in string_cols if c in df.columns]

    for col in existing_cols:
        # regex replace \u200b with empty string
        # cast to String first to handle Null type columns (all nulls)
        lf = lf.with_columns(pl.col(col).cast(pl.String).str.replace_all(r"[\u200b]", ""))

    # 2. Base Procedure ID
    # Regex extract EMEA/H/C/(\d+)
    lf = lf.with_columns(pl.col("product_number").str.extract(r"EMEA/H/C/(\d+)", 1).alias("base_procedure_id"))

    # 3. Substance Normalization (Split to List)
    # Replace + with / first then split
    if "active_substance" in df.columns:
        lf = lf.with_columns(
            pl.col("active_substance")
            .cast(pl.String)
            .str.replace_all(r"\+", "/")
            .str.split("/")
            .list.eval(pl.element().str.strip_chars())
            .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))  # Filter empty strings
            .alias("active_substance_list")
        )
    else:
        lf = lf.with_columns(pl.lit(None, dtype=pl.List(pl.String)).alias("active_substance_list"))

    # 4. ATC Code Explosion
    # Validate format (L7 standard): Letter, 2 Digits, 2 Letters, 2 Digits (e.g. A01BC01)
    if "atc_code" in df.columns:
        lf = lf.with_columns(
            pl.col("atc_code")
            .cast(pl.String)
            .str.to_uppercase()
            .str.replace_all(r",", ";")
            .str.split(";")
            .list.eval(pl.element().str.strip_chars())
            .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))  # Filter empty strings
            .list.eval(
                pl.element().filter(pl.element().str.contains(r"^[A-Z]\d{2}[A-Z]{2}\d{2}$"))
            )  # Strict L7 Validation
            .alias("atc_code_list")
        )
    else:
        lf = lf.with_columns(pl.lit(None, dtype=pl.List(pl.String)).alias("atc_code_list"))

    # 5. Status Standardization
    if "authorisation_status" in df.columns:
        lf = lf.with_columns(get_status_normalization_expr("authorisation_status").alias("status_normalized"))
    else:
        lf = lf.with_columns(pl.lit(None, dtype=pl.String).alias("status_normalized"))

    return lf.collect()


def generate_row_hash(df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
    """
    Generates an MD5 hash of the specified columns for each row.
    Adds a 'row_hash' column.
    """
    exprs = []
    schema = df.schema
    for c in columns:
        dtype = schema.get(c)
        if dtype is None:
            # Fallback for missing columns (shouldn't happen if validated)
            exprs.append(pl.lit(""))
        elif isinstance(dtype, pl.List):
            # List types cannot be cast to String directly in recent Polars
            # Join with semicolon to create string repr
            # Handle nulls inside list? .list.join ignores nulls usually or joins them?
            # We want a deterministic string.
            # .list.join returns String. fill_null handled after.
            exprs.append(pl.col(c).list.join(";").fill_null(""))
        else:
            exprs.append(pl.col(c).cast(pl.String).fill_null(""))

    expr = pl.concat_str(exprs, separator="|")

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
        current_snapshot: The new data (Cleaned Bronze)
        history: The existing history (Silver). Schema must include:
                 [primary_key, ..., valid_from, valid_to, is_current, row_hash]
        primary_key: The column name for the join key.
        ingestion_ts: The timestamp for valid_from/valid_to.
        hash_columns: Columns used to detect changes.

    Returns:
        Updated history DataFrame.
    """

    # 1. Prepare Snapshot
    # Deduplicate snapshot on primary key to prevent history corruption
    # We keep the first occurrence arbitrarily if duplicates exist
    snapshot_unique = current_snapshot.unique(subset=[primary_key], keep="first")
    snapshot_hashed = generate_row_hash(snapshot_unique, hash_columns)

    # 2. Identify Changes
    if history.is_empty():
        result = snapshot_hashed.with_columns(
            valid_from=pl.lit(ingestion_ts), valid_to=pl.lit(None, dtype=pl.Datetime), is_current=pl.lit(True)
        )
        # Bootstrap schema if history has no columns
        if len(history.columns) > 0:
            # Enforce exact schema types from history to prevent 'Null' vs 'String' issues
            return result.select([pl.col(c).cast(history.schema[c]) for c in history.columns])
        return result

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

    # Ensure output schema matches history schema (ignore extra columns from snapshot)
    return final_history.select([pl.col(c).cast(history.schema[c]) for c in history.columns])
