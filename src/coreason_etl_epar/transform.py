# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_epar

import polars as pl

from coreason_etl_epar.schemas import RegulatoryStatusEnum


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
    return standardize_authorisation_status(df)
