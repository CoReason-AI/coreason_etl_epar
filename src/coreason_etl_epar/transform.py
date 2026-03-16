# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_epar

import uuid
from datetime import datetime

import polars as pl

from coreason_etl_epar.schemas import RegulatoryStatusEnum

NAMESPACE_EMA = uuid.uuid5(uuid.NAMESPACE_DNS, "ema.europa.eu")


def generate_coreason_id(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    """
    AGENT INSTRUCTION: Generate coreason_id using UUID5.
    Logic: UUID5(NAMESPACE_EMA, source_id). source_id is product_number.
    """

    # We use a python function mapping or expression with string operations.
    # The most robust way in Polars is to use map_elements or create a custom python function,
    # but since uuid5 takes strings, we can use map_elements.

    def _uuid5_mapper(val: str | None) -> str | None:
        if val is None:
            return None  # pragma: no cover
        return str(uuid.uuid5(NAMESPACE_EMA, val))

    return df.with_columns(
        pl.col("product_number").map_elements(_uuid5_mapper, return_dtype=pl.String).alias("coreason_id")
    )


def normalize_base_procedure_id(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    """
    AGENT INSTRUCTION: Extract base_procedure_id from product_number.
    Logic: Regex extract EMEA/H/C/(\\d+) from product_number.
    """
    return df.with_columns(pl.col("product_number").str.extract(r"EMEA/H/C/(\d+)", 1).alias("base_procedure_id"))


def normalize_active_substance(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    """
    AGENT INSTRUCTION: Normalize active_substance.
    Logic: active_substance.str.split(by=["/", "+"]). Trim whitespace. Store as Array.
    """
    # Polars str.split requires string exact match, we need to use replace to standardize separators first
    return df.with_columns(
        pl.col("active_substance")
        .str.replace_all(r"\+", "/")
        .str.split("/")
        .list.eval(pl.element().str.strip_chars())
        .alias("active_substance")
    )


def normalize_atc_code(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    """
    AGENT INSTRUCTION: Explode ATC Code.
    Logic: atc_code.str.split(by=[";", ","]). Validate format (L7 standard). Store as Array.
    Trim whitespace and ignore empty elements.
    """
    # Standardize separator first, then split
    return df.with_columns(
        pl.when(pl.col("atc_code").is_not_null())
        .then(
            pl.col("atc_code")
            .str.replace_all(",", ";")
            .str.split(";")
            .list.eval(pl.element().str.strip_chars())
            .list.eval(pl.element().filter(pl.element().str.len_bytes() > 0))
        )
        .otherwise(pl.lit(None))
        .alias("atc_code")
    )


def standardize_authorisation_status(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    """
    AGENT INSTRUCTION: Standardize status.
    Authorised -> APPROVED
    Conditional -> CONDITIONAL_APPROVAL
    Exceptional Circumstances -> EXCEPTIONAL_CIRCUMSTANCES
    Refused -> REJECTED
    Withdrawn -> WITHDRAWN
    Suspended -> SUSPENDED
    """
    mapping = {
        "Authorised": RegulatoryStatusEnum.APPROVED.value,
        "Conditional": RegulatoryStatusEnum.CONDITIONAL_APPROVAL.value,
        "Exceptional Circumstances": RegulatoryStatusEnum.EXCEPTIONAL_CIRCUMSTANCES.value,
        "Refused": RegulatoryStatusEnum.REJECTED.value,
        "Withdrawn": RegulatoryStatusEnum.WITHDRAWN.value,
        "Suspended": RegulatoryStatusEnum.SUSPENDED.value,
    }

    # Use replace_strict instead of deprecated replace
    return df.with_columns(
        pl.col("authorisation_status")
        .replace_strict(mapping, default=pl.col("authorisation_status"))
        .alias("authorisation_status")
    )


def normalize_epar_fields(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    """
    AGENT INSTRUCTION: Pipeline to normalize all basic fields in the Silver layer.
    """
    df = normalize_base_procedure_id(df)
    df = normalize_active_substance(df)
    df = normalize_atc_code(df)
    df = standardize_authorisation_status(df)
    return generate_coreason_id(df)


def apply_scd_type_2(
    current_df: pl.LazyFrame | pl.DataFrame,
    new_snapshot: pl.LazyFrame | pl.DataFrame,
    ingestion_ts: datetime,
    id_col: str,
    hash_cols: list[str],
) -> pl.LazyFrame | pl.DataFrame:
    """
    AGENT INSTRUCTION: Implement SCD Type 2 logic to track history.
    """
    is_lazy = isinstance(current_df, pl.LazyFrame)
    curr: pl.LazyFrame = current_df if isinstance(current_df, pl.LazyFrame) else pl.LazyFrame(current_df)
    new: pl.LazyFrame = new_snapshot if isinstance(new_snapshot, pl.LazyFrame) else pl.LazyFrame(new_snapshot)

    # Resolve schemas to get column names safely
    new_cols_schema = new.collect_schema().names()
    curr_cols_schema = curr.collect_schema().names()

    # 1. Generate hashes for new snapshot
    new = new.with_columns(
        [
            pl.concat_str([pl.col(c).cast(pl.String) for c in hash_cols], separator="|")
            .hash(seed=42)
            .cast(pl.String)
            .alias("row_hash")
        ]
    )

    # 2. Generate hashes for current
    curr = curr.with_columns(
        [
            pl.concat_str([pl.col(c).cast(pl.String) for c in hash_cols], separator="|")
            .hash(seed=42)
            .cast(pl.String)
            .alias("row_hash")
        ]
    )

    # 3. Separate active and inactive current records
    curr_active = curr.filter(pl.col("is_current") == True)  # noqa: E712
    curr_inactive = curr.filter(pl.col("is_current") == False)  # noqa: E712

    # 4. Join active current records with new snapshot
    joined = curr_active.join(new, on=id_col, how="full", suffix="_new")

    # The joined schema will have `_new` suffixes for overlapping columns
    joined_cols_schema = joined.collect_schema().names()

    # Base columns from new snapshot (without hash)
    new_cols = [c for c in new_cols_schema if c != "row_hash"]
    # Base columns from current (without hash)
    curr_cols = [c for c in curr_cols_schema if c != "row_hash"]

    # Select expressions for pulling new row data from the joined frame
    select_exprs_for_new = [
        pl.col(f"{c}_new").alias(c) if f"{c}_new" in joined_cols_schema else pl.col(c) for c in new_cols
    ]

    # --- 1. New records (Insert) ---
    inserts = joined.filter(pl.col("row_hash").is_null() & pl.col("row_hash_new").is_not_null())
    inserts = inserts.select(select_exprs_for_new).with_columns(
        [
            pl.lit(ingestion_ts).alias("valid_from"),
            pl.lit(None, dtype=pl.Datetime).alias("valid_to"),
            pl.lit(True).alias("is_current"),
        ]
    )

    # --- 2. Updated records (Close old, Insert new) ---
    updates = joined.filter(
        pl.col("row_hash").is_not_null()
        & pl.col("row_hash_new").is_not_null()
        & (pl.col("row_hash") != pl.col("row_hash_new"))
    )

    closed_updates = updates.select(curr_cols).with_columns(
        [pl.lit(ingestion_ts).alias("valid_to"), pl.lit(False).alias("is_current")]
    )

    inserted_updates = updates.select(select_exprs_for_new).with_columns(
        [
            pl.lit(ingestion_ts).alias("valid_from"),
            pl.lit(None, dtype=pl.Datetime).alias("valid_to"),
            pl.lit(True).alias("is_current"),
        ]
    )

    # --- 3. Deleted/Vanished records (Close old) ---
    deletes = joined.filter(pl.col("row_hash").is_not_null() & pl.col("row_hash_new").is_null())
    closed_deletes = deletes.select(curr_cols).with_columns(
        [pl.lit(ingestion_ts).alias("valid_to"), pl.lit(False).alias("is_current")]
    )

    # --- 4. Unchanged records (Keep as is) ---
    unchanged = joined.filter(
        pl.col("row_hash").is_not_null()
        & pl.col("row_hash_new").is_not_null()
        & (pl.col("row_hash") == pl.col("row_hash_new"))
    ).select(curr_cols)

    # Combine everything
    result: pl.LazyFrame = pl.concat(
        [
            curr_inactive.select(curr_cols),
            unchanged,
            closed_updates,
            closed_deletes,
            inserts,
            inserted_updates,
        ],
        how="vertical_relaxed",
    )

    return result if is_lazy else result.collect()
