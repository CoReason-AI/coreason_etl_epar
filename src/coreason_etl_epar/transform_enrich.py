from typing import Dict

import polars as pl

from coreason_etl_epar.logger import logger


def jaro_winkler(s1: str, s2: str) -> float:
    """
    Pure Python implementation of Jaro-Winkler distance.
    """
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = (max(len1, len2) // 2) - 1

    matches = 0
    transpositions = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)

        for j in range(start, end):
            # Check if s2 char is already used
            if s2_matches[j]:
                continue  # pragma: no cover
            # Check for mismatch
            if s1[i] != s2[j]:
                continue

            s1_matches[i] = True  # pragma: no cover
            s2_matches[j] = True
            matches += 1
            break  # pragma: no cover

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    transpositions //= 2

    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0

    # Winkler modification
    prefix = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1.0 - jaro)


def enrich_epar(df: pl.DataFrame, spor_df: pl.DataFrame) -> pl.DataFrame:
    """
    Applies SPOR enrichment to EPAR dataframe using Lazy API and Streaming.
    Assumes `df` is already cleaned and normalized.

    Args:
        df: Cleaned Silver EPAR dataframe.
        spor_df: Silver SPOR Organisations dataframe (must have 'name', 'org_id').

    Returns:
        Enriched DataFrame with 'spor_mah_id'.
    """
    # Convert to LazyFrame
    lf = df.lazy()
    spor_lf = spor_df.lazy()

    # NOTE: Normalization logic (Base Procedure ID, Substance, ATC, Status)
    # moved to clean_epar_bronze in transform_silver.py

    # 5. Organization Enrichment (Fuzzy Join)
    # We need to map 'marketing_authorisation_holder' to 'spor_org_id'.
    # Strategy:
    # Get unique MAHs from EPAR
    mah_names_lf = lf.select("marketing_authorisation_holder").unique()

    # If SPOR is empty, return with null id
    # We need to check if spor_df is empty. Since we have the DF, we can check directly.
    if spor_df.is_empty():
        return lf.with_columns(pl.lit(None, dtype=pl.String).alias("spor_mah_id")).collect(engine="streaming")

    # Rename for clarity
    # SPOR DF expected cols: name, org_id
    spor_renamed_lf = spor_lf.select([pl.col("name").alias("spor_name"), pl.col("org_id").alias("spor_id")])

    # Cartesian Product
    cross_lf = mah_names_lf.join(spor_renamed_lf, how="cross")

    # Compute Distance
    # Use map_elements with jaro_winkler
    def calc_dist(struct: Dict[str, str]) -> float:
        return jaro_winkler(struct["marketing_authorisation_holder"].lower(), struct["spor_name"].lower())

    cross_lf = cross_lf.with_columns(
        pl.struct(["marketing_authorisation_holder", "spor_name"])
        .map_elements(calc_dist, return_dtype=pl.Float64)
        .alias("score")
    )

    # Filter > 0.90 and pick best match
    matches_lf = (
        cross_lf.filter(pl.col("score") > 0.90)
        .sort(["score", "spor_id"], descending=[True, False])
        .unique(subset=["marketing_authorisation_holder"], keep="first")
        .select(["marketing_authorisation_holder", "spor_id"])
    )

    # To calculate metrics, we must materialize the matches (or at least the count)
    # We collect the matches DF. It should be small (distinct MAHs).
    # Collecting here allows us to log the metric and then use the DF for joining.
    matches_df = (
        matches_lf.collect()
    )  # Not streaming here as it involves cross-join which might be tricky, but let's try strict default

    # Calculate & Log Match Rate Metric
    # We need total unique MAHs count.
    total_mah = mah_names_lf.collect().height
    matched_mah = matches_df.height
    match_rate = (matched_mah / total_mah) if total_mah > 0 else 0.0

    logger.bind(spor_match_rate=match_rate, metric="spor_match_rate").info(
        f"SPOR Match Rate: {match_rate:.2%} ({matched_mah}/{total_mah})"
    )

    if match_rate < 0.90 and total_mah > 0:
        logger.warning(f"SPOR Match Rate is below threshold: {match_rate:.2%}")

    # Join back to main DF
    # We use the materialized matches_df but convert to lazy for the join
    lf = lf.join(matches_df.lazy(), on="marketing_authorisation_holder", how="left").rename({"spor_id": "spor_mah_id"})

    # Return collected result with streaming
    # Note: Sort by product_number to ensure deterministic order (streaming might shuffle)
    if "product_number" in df.columns:
        lf = lf.sort("product_number")

    return lf.collect(engine="streaming")
