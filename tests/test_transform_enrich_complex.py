# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_epar

import polars as pl
import pytest

from coreason_etl_epar.transform_enrich import enrich_epar


@pytest.fixture
def base_epar_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "product_number": ["EMEA/H/C/001234"],
            "medicine_name": ["Med A"],
            "marketing_authorisation_holder": ["Pharma Corp"],
            "active_substance": ["Substance A"],
            "atc_code": ["A01BC01"],
            "authorisation_status": ["Authorised"],
            "url": ["http://example.com"],
        }
    )


def test_enrich_tie_breaking(base_epar_df: pl.DataFrame) -> None:
    """
    Complex Case: Two SPOR organizations match with identical scores (exact name match).
    Logic must strictly sort by Score DESC, then SPOR ID DESC (or ASC) to be deterministic.
    Code sorts by: ["score", "spor_id"], descending=[True, False] -> Score DESC, SPOR ID ASC.
    So strict match should pick the one with the 'smaller' (lexicographically) ID.
    """
    spor_df = pl.DataFrame(
        {
            "name": ["Pharma Corp", "Pharma Corp"],  # Identical Names = Identical Score
            "org_id": ["ORG-200", "ORG-100"],
            "roles": [["Marketing Authorisation Holder"], ["Marketing Authorisation Holder"]],
        }
    )

    result = enrich_epar(base_epar_df, spor_df)

    # Expectation: Pick ORG-100 because 'ORG-100' < 'ORG-200'
    assert result["spor_mah_id"].item() == "ORG-100"


def test_enrich_substance_parsing_complex(base_epar_df: pl.DataFrame) -> None:
    """
    Complex Case: Substance string with mixed delimiters (+ and /), empty segments, and whitespace.
    """
    # "Substance A + Substance B /  Substance C  / / + "
    # -> Split by + or / -> "Substance A ", " Substance B ", "  Substance C  ", " ", " ", " "
    # -> Strip -> "Substance A", "Substance B", "Substance C", "", "", ""
    # -> Filter -> "Substance A", "Substance B", "Substance C"
    df = base_epar_df.with_columns(active_substance=pl.lit("Substance A + Substance B /  Substance C  / / + "))
    spor_df = pl.DataFrame(schema={"name": pl.String, "org_id": pl.String})

    result = enrich_epar(df, spor_df)

    substances = result["active_substance_list"].item()
    # Sort for deterministic assertion
    assert sorted(substances) == ["Substance A", "Substance B", "Substance C"]
    assert "" not in substances


def test_enrich_atc_validation_complex(base_epar_df: pl.DataFrame) -> None:
    """
    Complex Case: ATC codes containing invalid formats, lowercase, and garbage.
    Valid: Letter, 2 Digits, 2 Letters, 2 Digits (e.g. A01BC01)
    """
    # a01bc01 = Valid (lowercase, should be normalized)
    # A01BC01 = Valid
    # XYZ = Invalid
    # A01BC012 = Invalid (too long)
    # A01BC = Invalid (too short)
    # B02BD02 = Valid
    # ;; = Empty segments
    input_str = "a01bc01, A01BC01, XYZ, A01BC012, B02BD02, , ;;"

    df = base_epar_df.with_columns(atc_code=pl.lit(input_str))
    spor_df = pl.DataFrame(schema={"name": pl.String, "org_id": pl.String})

    result = enrich_epar(df, spor_df)

    # .item() on a List column might return a Series in some Polars versions or if not cast?
    # Safer to use .to_list()[0] which guarantees Python list of lists -> take first.
    atc_list = result["atc_code_list"].to_list()[0]
    assert "A01BC01" in atc_list
    assert "B02BD02" in atc_list
    assert "a01bc01" not in atc_list  # Should be normalized to A01BC01
    assert "XYZ" not in atc_list
    assert "" not in atc_list
    # A01BC01 appears twice in input (one lower, one upper). Uniqueness not strictly required by FRD for this list,
    # but standard usually keeps them. If unique is applied later, that's fine.
    # Logic: split -> list.
    # a01bc01 -> A01BC01.
    # So A01BC01 should appear twice if list.unique() is not called.
    # Let's check count.
    assert atc_list.count("A01BC01") == 2


def test_enrich_empty_spor(base_epar_df: pl.DataFrame) -> None:
    """
    Edge Case: SPOR dataframe is completely empty.
    Should not crash, spor_mah_id should be null.
    """
    spor_df = pl.DataFrame(schema={"name": pl.String, "org_id": pl.String})

    result = enrich_epar(base_epar_df, spor_df)

    assert result["spor_mah_id"].item() is None
    assert "active_substance_list" in result.columns


def test_enrich_null_inputs(base_epar_df: pl.DataFrame) -> None:
    """
    Edge Case: Null values in active_substance or atc_code.
    """
    df = base_epar_df.with_columns(
        active_substance=pl.lit(None, dtype=pl.String), atc_code=pl.lit(None, dtype=pl.String)
    )
    spor_df = pl.DataFrame(schema={"name": pl.String, "org_id": pl.String})

    result = enrich_epar(df, spor_df)

    # Check that it didn't crash and returns nulls
    assert result["active_substance_list"].item() is None
    assert result["atc_code_list"].item() is None
