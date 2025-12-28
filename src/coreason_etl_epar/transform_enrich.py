import polars as pl
import re
from typing import List, Optional

def normalize_status(status: str) -> str:
    """
    Standardizes Authorisation Status to Enum values.
    """
    s = status.strip().upper()
    if "AUTHORISED" in s:
        return "APPROVED"
    if "CONDITIONAL" in s:
        return "CONDITIONAL_APPROVAL"
    if "EXCEPTIONAL" in s:
        return "EXCEPTIONAL_CIRCUMSTANCES"
    if "REFUSED" in s:
        return "REJECTED"
    if "WITHDRAWN" in s:
        return "WITHDRAWN"
    if "SUSPENDED" in s:
        return "SUSPENDED"
    return "UNKNOWN" # Fallback

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
                continue # pragma: no cover
            # Check for mismatch
            if s1[i] != s2[j]:
                continue

            s1_matches[i] = True # pragma: no cover
            s2_matches[j] = True
            matches += 1
            break # pragma: no cover

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
    Applies cleaning and enrichment to EPAR dataframe.

    Args:
        df: Silver EPAR dataframe.
        spor_df: Silver SPOR Organisations dataframe (must have 'name', 'org_id').

    Returns:
        Enriched DataFrame with cleaned fields and spor_mah_id.
    """

    # 1. Base Procedure ID
    # Regex extract EMEA/H/C/(\d+)
    df = df.with_columns(
        pl.col("product_number").str.extract(r"EMEA/H/C/(\d+)", 1).alias("base_procedure_id")
    )

    # 2. Substance Normalization (Split to List)
    # Refined Substance: Replace + with / first then split
    df = df.with_columns(
        pl.col("active_substance")
        .cast(pl.String)
        .str.replace_all(r"\+", "/")
        .str.split("/")
        .list.eval(pl.element().str.strip_chars())
        .alias("active_substance_list")
    )

    # 3. ATC Code Explosion
    df = df.with_columns(
        pl.col("atc_code")
        .cast(pl.String)
        .str.replace_all(r",", ";")
        .str.split(";")
        .list.eval(pl.element().str.strip_chars())
        .alias("atc_code_list")
    )

    # 4. Status Standardization
    df = df.with_columns(
        pl.col("authorisation_status").map_elements(normalize_status, return_dtype=pl.String).alias("status_normalized")
    )

    # 5. Organization Enrichment (Fuzzy Join)
    # We need to map 'marketing_authorisation_holder' to 'spor_org_id'.
    # Strategy:
    # Get unique MAHs from EPAR
    mah_names = df.select("marketing_authorisation_holder").unique()

    # If SPOR is empty, return with null id
    if spor_df.is_empty():
        return df.with_columns(pl.lit(None, dtype=pl.String).alias("spor_mah_id"))

    # Cross join unique MAHs with SPOR Names
    # Note: Optimization - only join if we have data.

    # Rename for clarity
    # SPOR DF expected cols: name, org_id
    spor_renamed = spor_df.select([pl.col("name").alias("spor_name"), pl.col("org_id").alias("spor_id")])

    # Cartesian Product
    cross = mah_names.join(spor_renamed, how="cross")

    # Compute Distance
    # Use map_elements with jaro_winkler
    # This is expensive, so we do it on unique names (small set)

    def calc_dist(struct):
        return jaro_winkler(struct["marketing_authorisation_holder"].lower(), struct["spor_name"].lower())

    cross = cross.with_columns(
        pl.struct(["marketing_authorisation_holder", "spor_name"])
        .map_elements(calc_dist, return_dtype=pl.Float64)
        .alias("score")
    )

    # Filter > 0.90 and pick best match
    matches = (
        cross
        .filter(pl.col("score") > 0.90)
        .sort("score", descending=True)
        .unique(subset=["marketing_authorisation_holder"], keep="first")
        .select(["marketing_authorisation_holder", "spor_id"])
    )

    # Join back to main DF
    df = df.join(matches, on="marketing_authorisation_holder", how="left").rename({"spor_id": "spor_mah_id"})

    return df
