import uuid
from datetime import datetime

import polars as pl
from polars.testing import assert_frame_equal

from coreason_etl_epar.schemas import RegulatoryStatusEnum
from coreason_etl_epar.transform import (
    NAMESPACE_EMA,
    _jaro_winkler_distance,
    apply_scd_type_2,
    enrich_organizations,
    generate_coreason_id,
    normalize_active_substance,
    normalize_atc_code,
    normalize_base_procedure_id,
    normalize_epar_fields,
    standardize_authorisation_status,
)


def test_normalize_base_procedure_id() -> None:
    df = pl.DataFrame({"product_number": ["EMEA/H/C/001234", "EMEA/H/C/000001", "invalid", None]})
    expected = pl.DataFrame(
        {
            "product_number": ["EMEA/H/C/001234", "EMEA/H/C/000001", "invalid", None],
            "base_procedure_id": ["001234", "000001", None, None],
        }
    )
    result = normalize_base_procedure_id(df)
    assert_frame_equal(result, expected)


def test_normalize_active_substance() -> None:
    df = pl.DataFrame(
        {"active_substance": ["substance1 / substance2", "substanceA+substanceB", "substance_only", None]}
    )
    expected = pl.DataFrame(
        {
            "active_substance": [
                ["substance1", "substance2"],
                ["substanceA", "substanceB"],
                ["substance_only"],
                None,
            ]
        }
    )
    result = normalize_active_substance(df)
    assert_frame_equal(result, expected)


def test_normalize_atc_code() -> None:
    df = pl.DataFrame({"atc_code": ["A01B; C02D", "D03E,F04G", "   ", None, "A01B; ; C02D"]})
    expected = pl.DataFrame(
        {
            "atc_code": [
                ["A01B", "C02D"],
                ["D03E", "F04G"],
                [],
                None,
                ["A01B", "C02D"],
            ]
        }
    )
    result = normalize_atc_code(df)
    assert_frame_equal(result, expected)


def test_standardize_authorisation_status() -> None:
    df = pl.DataFrame(
        {
            "authorisation_status": [
                "Authorised",
                "Conditional",
                "Exceptional Circumstances",
                "Refused",
                "Withdrawn",
                "Suspended",
                "Unknown",
                None,
            ]
        }
    )
    expected = pl.DataFrame(
        {
            "authorisation_status": [
                RegulatoryStatusEnum.APPROVED.value,
                RegulatoryStatusEnum.CONDITIONAL_APPROVAL.value,
                RegulatoryStatusEnum.EXCEPTIONAL_CIRCUMSTANCES.value,
                RegulatoryStatusEnum.REJECTED.value,
                RegulatoryStatusEnum.WITHDRAWN.value,
                RegulatoryStatusEnum.SUSPENDED.value,
                "Unknown",
                None,
            ]
        }
    )
    result = standardize_authorisation_status(df)
    assert_frame_equal(result, expected)


def test_generate_coreason_id() -> None:
    df = pl.DataFrame({"product_number": ["EMEA/H/C/001234", "EMEA/H/C/000001", None]})

    expected_id_1 = str(uuid.uuid5(NAMESPACE_EMA, "EMEA/H/C/001234"))
    expected_id_2 = str(uuid.uuid5(NAMESPACE_EMA, "EMEA/H/C/000001"))

    expected = pl.DataFrame(
        {
            "product_number": ["EMEA/H/C/001234", "EMEA/H/C/000001", None],
            "coreason_id": [expected_id_1, expected_id_2, None],
        }
    )
    result = generate_coreason_id(df)
    assert_frame_equal(result, expected)


def test_normalize_epar_fields() -> None:
    df = pl.DataFrame(
        {
            "product_number": ["EMEA/H/C/001234"],
            "active_substance": ["substance1 / substance2"],
            "atc_code": ["A01B; C02D"],
            "authorisation_status": ["Authorised"],
        }
    )

    expected_coreason_id = str(uuid.uuid5(NAMESPACE_EMA, "EMEA/H/C/001234"))

    expected = pl.DataFrame(
        {
            "product_number": ["EMEA/H/C/001234"],
            "active_substance": [["substance1", "substance2"]],
            "atc_code": [["A01B", "C02D"]],
            "authorisation_status": [RegulatoryStatusEnum.APPROVED.value],
            "base_procedure_id": ["001234"],
            "coreason_id": [expected_coreason_id],
        }
    )

    # We need to test the returned DataFrame by column as `map_elements` may change ordering
    result = normalize_epar_fields(df)

    for col in expected.columns:
        assert_frame_equal(result.select(col), expected.select(col))


def test_apply_scd_type_2() -> None:
    current_df = pl.DataFrame(
        {
            "source_id": ["A", "B", "C"],
            "col1": ["valA", "valB", "valC"],
            "col2": [1, 2, 3],
            "valid_from": [datetime(2023, 1, 1), datetime(2023, 1, 1), datetime(2023, 1, 1)],
            "valid_to": [None, None, None],
            "is_current": [True, True, True],
        }
    )

    # A: Unchanged
    # B: Updated (col2 changed)
    # C: Vanished (not in new snapshot)
    # D: Inserted (new in snapshot)
    new_snapshot = pl.DataFrame(
        {
            "source_id": ["A", "B", "D"],
            "col1": ["valA", "valB", "valD"],
            "col2": [1, 22, 4],
        }
    )

    ingestion_ts = datetime(2023, 2, 1)

    result = apply_scd_type_2(
        current_df=current_df,
        new_snapshot=new_snapshot,
        ingestion_ts=ingestion_ts,
        id_col="source_id",
        hash_cols=["col1", "col2"],
    )

    assert isinstance(result, pl.DataFrame)

    # We expect 5 rows:
    # 1. A unchanged (is_current=True)
    # 2. B old (is_current=False, valid_to=ingestion_ts)
    # 3. B new (is_current=True, valid_from=ingestion_ts)
    # 4. C vanished (is_current=False, valid_to=ingestion_ts)
    # 5. D inserted (is_current=True, valid_from=ingestion_ts)

    # Check A
    row_a = result.filter(pl.col("source_id") == "A")
    assert len(row_a) == 1
    assert row_a["is_current"][0]
    assert row_a["valid_to"][0] is None

    # Check B
    row_b = result.filter(pl.col("source_id") == "B").sort("valid_from")
    assert len(row_b) == 2
    assert not row_b["is_current"][0]
    assert row_b["valid_to"][0] == ingestion_ts
    assert row_b["is_current"][1]
    assert row_b["valid_from"][1] == ingestion_ts
    assert row_b["col2"][1] == 22

    # Check C
    row_c = result.filter(pl.col("source_id") == "C")
    assert len(row_c) == 1
    assert not row_c["is_current"][0]
    assert row_c["valid_to"][0] == ingestion_ts

    # Check D
    row_d = result.filter(pl.col("source_id") == "D")
    assert len(row_d) == 1
    assert row_d["is_current"][0]
    assert row_d["valid_from"][0] == ingestion_ts
    assert row_d["col2"][0] == 4


def test_jaro_winkler_distance() -> None:
    # Exact match
    assert _jaro_winkler_distance("hello", "hello") == 1.0

    # No match
    assert _jaro_winkler_distance("abc", "xyz") == 0.0

    # Empty string
    assert _jaro_winkler_distance("", "a") == 0.0
    assert _jaro_winkler_distance("a", "") == 0.0
    assert _jaro_winkler_distance("", "") == 1.0

    # Fuzzy match tests (expected > 0.90 but < 1.0)
    # E.g. minor typos
    score1 = _jaro_winkler_distance("mcdonalds", "macdonalds")
    assert 0.90 < score1 < 1.0

    score2 = _jaro_winkler_distance("pfizer inc", "pfizer inc.")
    assert 0.90 < score2 < 1.0

    score3 = _jaro_winkler_distance("bayer", "bayer ag")
    assert 0.90 < score3 < 1.0

    # Low match (expected <= 0.90)
    score4 = _jaro_winkler_distance("novartis", "pfizer")
    assert score4 <= 0.90


def test_enrich_organizations() -> None:
    epar_df = pl.DataFrame(
        {
            "product_number": ["1", "2", "3", "4", "5"],
            "marketing_authorisation_holder": [
                "Pfizer Inc.",  # Exact match except case
                "Bayer AG",  # Exact match
                "Novartiss",  # Typo match (should be > 0.90 fuzzy match)
                "Some Random Company",  # No match
                None,  # Null check
            ],
        }
    )

    spor_df = pl.DataFrame(
        {
            "org_id": ["ORG01", "ORG02", "ORG03", "ORG04"],
            "org_name": [
                "pfizer inc.",
                "Bayer AG",
                "Novartis",
                None,  # Null org name check
            ],
        }
    )

    expected_df = pl.DataFrame(
        {
            "product_number": ["1", "2", "3", "4", "5"],
            "marketing_authorisation_holder": [
                "Pfizer Inc.",
                "Bayer AG",
                "Novartiss",
                "Some Random Company",
                None,
            ],
            "spor_mah_id": [
                "ORG01",  # pfizer inc.
                "ORG02",  # Bayer AG
                "ORG03",  # Novartis
                None,
                None,
            ],
        }
    )

    result_df = enrich_organizations(epar_df, spor_df, threshold=0.90)

    assert isinstance(result_df, pl.DataFrame)
    assert_frame_equal(result_df, expected_df)


def test_enrich_organizations_lazy() -> None:
    epar_df = pl.LazyFrame(
        {
            "product_number": ["1"],
            "marketing_authorisation_holder": ["Pfizer Inc."],
        }
    )
    spor_df = pl.LazyFrame(
        {
            "org_id": ["ORG01"],
            "org_name": ["pfizer inc."],
        }
    )

    result_df = enrich_organizations(epar_df, spor_df, threshold=0.90)
    assert isinstance(result_df, pl.LazyFrame)
    assert result_df.collect()["spor_mah_id"][0] == "ORG01"


def test_apply_scd_type_2_lazy() -> None:
    current_df = pl.LazyFrame(
        {
            "source_id": ["A"],
            "val": ["old"],
            "valid_from": [datetime(2023, 1, 1)],
            "valid_to": [None],
            "is_current": [True],
        }
    )

    new_snapshot = pl.LazyFrame(
        {
            "source_id": ["A"],
            "val": ["new"],
        }
    )

    result = apply_scd_type_2(
        current_df=current_df,
        new_snapshot=new_snapshot,
        ingestion_ts=datetime(2023, 2, 1),
        id_col="source_id",
        hash_cols=["val"],
    )

    assert isinstance(result, pl.LazyFrame)
    df = result.collect()
    assert len(df) == 2
