import polars as pl
from polars.testing import assert_frame_equal

from coreason_etl_epar.schemas import RegulatoryStatusEnum
from coreason_etl_epar.transform import (
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


def test_normalize_epar_fields() -> None:
    df = pl.DataFrame(
        {
            "product_number": ["EMEA/H/C/001234"],
            "active_substance": ["substance1 / substance2"],
            "atc_code": ["A01B; C02D"],
            "authorisation_status": ["Authorised"],
        }
    )
    expected = pl.DataFrame(
        {
            "product_number": ["EMEA/H/C/001234"],
            "active_substance": [["substance1", "substance2"]],
            "atc_code": [["A01B", "C02D"]],
            "authorisation_status": [RegulatoryStatusEnum.APPROVED.value],
            "base_procedure_id": ["001234"],
        }
    )
    result = normalize_epar_fields(df)
    assert_frame_equal(result, expected)
