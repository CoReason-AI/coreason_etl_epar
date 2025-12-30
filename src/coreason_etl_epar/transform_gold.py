import uuid
from typing import Dict

import polars as pl

# Constant Namespace for UUID5 generation
NAMESPACE_EMA = uuid.uuid5(uuid.NAMESPACE_DNS, "ema.europa.eu")


def generate_coreason_id(source_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE_EMA, source_id))


def create_gold_layer(silver_df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """
    Transforms Silver Data into Gold Star Schema tables.

    Args:
        silver_df: Enriched Silver DataFrame.

    Returns:
        Dictionary containing 'dim_medicine', 'fact_regulatory_history', 'bridge_medicine_features'
    """

    # 1. Generate Coreason ID
    df = silver_df.with_columns(
        pl.col("product_number").map_elements(generate_coreason_id, return_dtype=pl.String).alias("coreason_id")
    )

    # 2. dim_medicine (Immutable Entity Attributes)
    current_df = df.filter(pl.col("is_current"))
    if current_df.is_empty() and not df.is_empty():
        current_df = df.sort("valid_from", descending=True).unique(subset=["coreason_id"], keep="first")

    dim_medicine = current_df.select(
        [
            "coreason_id",
            "medicine_name",
            "base_procedure_id",
            pl.col("medicine_name").alias("brand_name"),
            pl.col("biosimilar").fill_null(False).alias("is_biosimilar"),
            pl.col("generic").fill_null(False).alias("is_generic"),
            pl.col("orphan").fill_null(False).alias("is_orphan"),
            pl.col("url").alias("ema_product_url"),
        ]
    )

    # 3. fact_regulatory_history (SCD Type 2 Timeline)
    fact_history = df.select(
        [
            pl.concat_str([pl.col("coreason_id"), pl.col("valid_from").cast(pl.String)], separator="_")
            .map_elements(lambda x: str(uuid.uuid5(uuid.NAMESPACE_OID, x)), return_dtype=pl.String)
            .alias("history_id"),
            "coreason_id",
            pl.col("status_normalized").alias("status"),
            "valid_from",
            "valid_to",
            "is_current",
            "spor_mah_id",
        ]
    )

    # 4. bridge_medicine_features
    # ATC
    atc = (
        current_df.select(["coreason_id", "atc_code_list"])
        .explode("atc_code_list")
        .drop_nulls()
        .select(
            ["coreason_id", pl.lit("ATC_CODE").alias("feature_type"), pl.col("atc_code_list").alias("feature_value")]
        )
    )

    # Substance
    substance = (
        current_df.select(["coreason_id", "active_substance_list"])
        .explode("active_substance_list")
        .drop_nulls()
        .select(
            [
                "coreason_id",
                pl.lit("SUBSTANCE").alias("feature_type"),
                pl.col("active_substance_list").alias("feature_value"),
            ]
        )
    )

    # Therapeutic Area
    area = (
        current_df.select(["coreason_id", "therapeutic_area"])
        .drop_nulls()
        .with_columns(pl.col("therapeutic_area").str.split(";").alias("area_list"))
        .explode("area_list")
        .select(
            [
                "coreason_id",
                pl.lit("THERAPEUTIC_AREA").alias("feature_type"),
                pl.col("area_list").str.strip_chars().alias("feature_value"),
            ]
        )
        .filter(pl.col("feature_value").str.len_chars() > 0)
    )

    bridge = pl.concat([atc, substance, area])

    return {"dim_medicine": dim_medicine, "fact_regulatory_history": fact_history, "bridge_medicine_features": bridge}
