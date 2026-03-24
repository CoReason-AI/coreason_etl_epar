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
from rapidfuzz.distance import JaroWinkler

from coreason_etl_epar.schemas import FeatureTypeEnum, RegulatoryStatusEnum
from coreason_etl_epar.utils.logger import logger

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
    # Standardize separator first, then split, trim, and filter by L7 standard regex: ^[A-Z]\d{2}[A-Z]{2}\d{2}$
    return df.with_columns(
        pl.when(pl.col("atc_code").is_not_null())
        .then(
            pl.col("atc_code")
            .str.replace_all(",", ";")
            .str.split(";")
            .list.eval(pl.element().str.strip_chars())
            .list.eval(
                pl.element().filter(
                    (pl.element().str.len_bytes() > 0) & (pl.element().str.contains(r"^[A-Z]\d{2}[A-Z]{2}\d{2}$"))
                )
            )
        )
        .otherwise(pl.lit(None))
        .alias("atc_code")
    )


def normalize_therapeutic_area(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    """
    AGENT INSTRUCTION: Extract Therapeutic Areas into a searchable array.
    Logic: therapeutic_area.str.split(by=[";", ","]). Trim whitespace and ignore empty elements.
    """
    return df.with_columns(
        pl.when(pl.col("therapeutic_area").is_not_null())
        .then(
            pl.col("therapeutic_area")
            .str.replace_all(",", ";")
            .str.split(";")
            .list.eval(pl.element().str.strip_chars())
            .list.eval(pl.element().filter(pl.element().str.len_bytes() > 0))
        )
        .otherwise(pl.lit(None))
        .alias("therapeutic_area")
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
    df = normalize_therapeutic_area(df)
    df = standardize_authorisation_status(df)
    return generate_coreason_id(df)


def _jaro_winkler_distance(s1: str, s2: str) -> float:
    """
    AGENT INSTRUCTION: Wrapper for rapidfuzz JaroWinkler distance.
    Returns similarity between 0.0 and 1.0.
    """
    return float(JaroWinkler.similarity(s1, s2))


def enrich_organizations(
    epar_df: pl.LazyFrame | pl.DataFrame,
    spor_df: pl.LazyFrame | pl.DataFrame,
    threshold: float = 0.90,
) -> pl.LazyFrame | pl.DataFrame:
    """
    AGENT INSTRUCTION: Organization Enrichment (Offline Join).
    Logic: Silver_EPAR LEFT JOIN Silver_SPOR_Orgs ON fuzzy_match(mah_name, org_name).
    Threshold: Jaro-Winkler distance > 0.90.
    """
    is_lazy = isinstance(epar_df, pl.LazyFrame)

    epar = epar_df.lazy() if not is_lazy else epar_df
    spor = spor_df.lazy() if not isinstance(spor_df, pl.LazyFrame) else spor_df

    # To maintain lazy evaluation and streaming compatibility, we avoid Python loops inside `map_elements`
    # over the entire SPOR list. Since Polars does not have native Jaro-Winkler in `pl.Expr.str`,
    # we implement the join lazily.
    # We will do a cross join but filter heavily to keep performance acceptable,
    # or since we are required to do "fuzzy_match" using python, we map over a tuple of arrays if needed.
    # However, standard practice without Polars native string metric is a cross join (or mapped cross join)
    # followed by python computation. To prevent exploding memory, we could use `.map_batches`.

    def _fuzzy_match_batch(s: pl.Series) -> pl.Series:
        return s  # pragma: no cover

    # Actually, a full cross-join is bad for memory. Let's do a cross-join on unique MAHs.

    # Extract unique MAHs from EPAR lazily
    unique_mahs = epar.select(pl.col("marketing_authorisation_holder").alias("mah_name")).unique().drop_nulls()

    # Cross join unique MAHs with all SPOR organizations
    spor_select = spor.select(pl.col("org_id"), pl.col("org_name").alias("spor_name")).drop_nulls()
    spor_select_lf = spor_select if isinstance(spor_select, pl.LazyFrame) else pl.LazyFrame(spor_select)
    unique_mahs_lf = unique_mahs if isinstance(unique_mahs, pl.LazyFrame) else pl.LazyFrame(unique_mahs)

    cross_joined = unique_mahs_lf.join(
        spor_select_lf,
        how="cross",
    )

    # Compute jaro winkler distance in python via map_batches for performance
    def _compute_jaro_batch(s: pl.Series) -> pl.Series:
        # s is a Series of Structs
        scores = []
        for row in s:
            m = row["mah_name"]
            spor_n = row["spor_name"]
            if m is None or spor_n is None:
                scores.append(0.0)  # pragma: no cover
            else:
                scores.append(_jaro_winkler_distance(m.lower(), spor_n.lower()))

        return pl.Series("score", scores, dtype=pl.Float64)

    # Apply batch computation
    scored = cross_joined.with_columns(
        pl.struct(["mah_name", "spor_name"]).map_batches(_compute_jaro_batch, return_dtype=pl.Float64).alias("score")
    )

    # Filter by threshold and find the best match
    filtered = scored.filter(pl.col("score") > threshold)

    # Sort to get the highest score first, then distinct by mah_name
    best_matches = filtered.sort(["mah_name", "score"], descending=[False, True]).unique(
        subset=["mah_name"], keep="first"
    )

    # Now join back to the original EPAR frame
    best_matches_selected = best_matches.select(
        [pl.col("mah_name").alias("marketing_authorisation_holder"), pl.col("org_id").alias("spor_mah_id")]
    )
    # Re-cast to correct types for mypy before join
    epar_lf = epar if isinstance(epar, pl.LazyFrame) else pl.LazyFrame(epar)
    best_matches_selected_lf = (
        best_matches_selected
        if isinstance(best_matches_selected, pl.LazyFrame)
        else pl.LazyFrame(best_matches_selected)
    )

    enriched = epar_lf.join(
        best_matches_selected_lf,
        on="marketing_authorisation_holder",
        how="left",
    )

    # Calculate spor_match_rate safely without collecting the entire frame if it's lazy
    stats_query = enriched.select(
        [
            pl.len().alias("total"),
            pl.col("spor_mah_id").drop_nulls().len().alias("matched"),
        ]
    )

    if isinstance(stats_query, pl.LazyFrame):
        # We try to use streaming if available to avoid loading everything in memory
        try:
            stats_df = stats_query.collect(engine="streaming")
        except Exception:  # pragma: no cover
            stats_df = stats_query.collect()
    else:
        stats_df = stats_query  # pragma: no cover

    stats = stats_df.row(0)
    total_mahs = stats[0]
    matched_mahs = stats[1]

    spor_match_rate = matched_mahs / total_mahs if total_mahs > 0 else 0.0  # pragma: no cover

    logger.info("SPOR match rate calculated", spor_match_rate=spor_match_rate)

    if spor_match_rate < 0.90:
        logger.warning(f"SPOR match rate is below 90%: {spor_match_rate:.2%}")

    return (
        enriched
        if is_lazy
        else enriched.collect(engine="streaming")
        if isinstance(enriched, pl.LazyFrame)
        else enriched
    )


def build_dim_medicine(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    """
    AGENT INSTRUCTION: Transform Silver data into the `dim_medicine` Gold table format.
    Extract immutable entity attributes and deduplicate by coreason_id.
    """
    return df.select(
        [
            pl.col("coreason_id"),
            pl.col("medicine_name"),
            pl.col("base_procedure_id"),
            pl.lit(None, dtype=pl.String).alias("brand_name"),
            pl.col("biosimilar").fill_null(False).alias("is_biosimilar"),
            pl.col("generic").fill_null(False).alias("is_generic"),
            pl.col("additional_monitoring").fill_null(False).alias("additional_monitoring"),
            pl.col("url").alias("ema_product_url"),
        ]
    ).unique(subset=["coreason_id"], keep="first")


def build_fact_regulatory_history(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    """
    AGENT INSTRUCTION: Transform Silver history data into the `fact_regulatory_history` Gold table format.
    """

    def _generate_history_id(s: pl.Series) -> pl.Series:
        # Generate a deterministic history ID based on coreason_id and valid_from
        ids: list[str | None] = []
        for row in s:
            c_id = row["coreason_id"]
            v_from = row["valid_from"]
            if c_id is None or v_from is None:
                ids.append(None)  # pragma: no cover
            else:
                # v_from is datetime, format it deterministically
                v_str = v_from.isoformat()
                combo = f"{c_id}|{v_str}"
                ids.append(str(uuid.uuid5(NAMESPACE_EMA, combo)))
        return pl.Series("history_id", ids, dtype=pl.String)

    is_lazy = isinstance(df, pl.LazyFrame)
    d = df.lazy() if not is_lazy else df

    # We need: history_id, coreason_id, status, valid_from, valid_to, is_current, spor_mah_id, is_orphan
    # Select existing columns and rename to match Gold schema
    select_cols = [
        pl.col("coreason_id"),
        pl.col("authorisation_status").alias("status"),
        pl.col("valid_from"),
        pl.col("valid_to"),
        pl.col("is_current"),
    ]

    schema_names = d.collect_schema().names()

    # Handle spor_mah_id which might not exist if enrichment wasn't done or match wasn't found
    if "spor_mah_id" in schema_names:
        select_cols.append(pl.col("spor_mah_id"))
    else:
        select_cols.append(pl.lit(None, dtype=pl.String).alias("spor_mah_id"))

    # Map orphan to is_orphan
    if "orphan" in schema_names:
        select_cols.append(pl.col("orphan").fill_null(False).alias("is_orphan"))
    else:
        select_cols.append(pl.lit(False, dtype=pl.Boolean).alias("is_orphan"))

    d = d.select(select_cols)

    # Generate history_id
    d = d.with_columns(
        pl.struct(["coreason_id", "valid_from"])
        .map_batches(_generate_history_id, return_dtype=pl.String)
        .alias("history_id")
    )

    # Reorder columns to match schema
    d = d.select(
        [
            pl.col("history_id"),
            pl.col("coreason_id"),
            pl.col("status"),
            pl.col("valid_from"),
            pl.col("valid_to"),
            pl.col("is_current"),
            pl.col("spor_mah_id"),
            pl.col("is_orphan"),
        ]
    )

    return d if is_lazy else d.collect() if isinstance(d, pl.LazyFrame) else d


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
    result = pl.concat(
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

    # Calculate scd_updates_count
    # To determine the number of distinct records that changed status/data today
    # We can calculate it directly from the `joined` frame, to avoid repeatedly evaluating lazy frames.
    changed_records_filter = (
        (pl.col("row_hash").is_null() & pl.col("row_hash_new").is_not_null())
        | (
            pl.col("row_hash").is_not_null()
            & pl.col("row_hash_new").is_not_null()
            & (pl.col("row_hash") != pl.col("row_hash_new"))
        )
        | (pl.col("row_hash").is_not_null() & pl.col("row_hash_new").is_null())
    )

    updates_count_query = joined.filter(changed_records_filter).select(pl.len().alias("count"))

    if isinstance(updates_count_query, pl.LazyFrame):
        try:
            updates_count_df = updates_count_query.collect(engine="streaming")
        except Exception:  # pragma: no cover
            updates_count_df = updates_count_query.collect()
    else:
        updates_count_df = updates_count_query  # pragma: no cover

    scd_updates_count = updates_count_df.row(0)[0]

    logger.info("SCD Type 2 applied", scd_updates_count=scd_updates_count)

    return result if is_lazy else result.collect() if isinstance(result, pl.LazyFrame) else result


def build_bridge_medicine_features(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    """
    AGENT INSTRUCTION: Transform Silver data into the `bridge_medicine_features` Gold table format.
    Un-nest array columns (active_substance, atc_code, therapeutic_area) and melt into EAV format.
    """
    is_lazy = isinstance(df, pl.LazyFrame)
    d = df.lazy() if not is_lazy else df

    schema_cols = d.collect_schema().names()

    # Map column names to their Gold enum representations
    feature_columns = {
        "active_substance": FeatureTypeEnum.SUBSTANCE.value,
        "atc_code": FeatureTypeEnum.ATC_CODE.value,
        "therapeutic_area": FeatureTypeEnum.THERAPEUTIC_AREA.value,
    }

    # Filter to only the columns that actually exist in the dataframe
    cols_to_melt = [col for col in feature_columns if col in schema_cols]

    if not cols_to_melt:
        # If no relevant columns exist, return an empty frame with correct schema
        empty_df = pl.LazyFrame(
            schema={
                "coreason_id": pl.String,
                "feature_type": pl.String,
                "feature_value": pl.String,
            }
        )
        return empty_df if is_lazy else empty_df.collect()

    # 1. Select the required base ID and the existing feature columns
    # 2. Unpivot (melt) the wide dataframe into a long EAV format
    # 3. Explode the list arrays to flatten values into distinct rows
    # 4. Map the melted variable names to standard Enums
    # 5. Drop empty/null values and deduplicate
    melted = (
        d.select(["coreason_id", *cols_to_melt])
        .unpivot(index="coreason_id", variable_name="feature_type", value_name="feature_value")
        .explode("feature_value")
        .drop_nulls(subset=["feature_value"])
        .with_columns(pl.col("feature_type").replace_strict(feature_columns).alias("feature_type"))
        .unique(subset=["coreason_id", "feature_type", "feature_value"])
    )

    return melted if is_lazy else melted.collect() if isinstance(melted, pl.LazyFrame) else melted
