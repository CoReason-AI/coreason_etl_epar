# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_epar

from datetime import datetime
from typing import Any, Dict

import polars as pl
import pytest
from pydantic import ValidationError

from coreason_etl_epar.schema import EPARSourceRow
from coreason_etl_epar.transform_silver import apply_scd2, generate_row_hash


def test_scd2_resurrection() -> None:
    """
    Edge Case: Resurrection.
    A product was active, then deleted (closed), and now reappears.
    Expected: A new active record is created. The old closed record remains closed.
    """
    ts_1 = datetime(2024, 1, 1)
    ts_2 = datetime(2024, 1, 2)  # Deletion time
    ts_3 = datetime(2024, 1, 3)  # Resurrection time

    schema: Dict[str, pl.DataType | Any] = {
        "id": pl.Int64,
        "data": pl.String,
        "valid_from": pl.Datetime,
        "valid_to": pl.Datetime,
        "is_current": pl.Boolean,
        "row_hash": pl.String,
    }

    # History: ID=1 was valid from ts_1 to ts_2, then closed (is_current=False)
    # It currently does NOT exist in the "active" view.
    history = pl.DataFrame(
        {
            "id": [1],
            "data": ["A"],
            "valid_from": [ts_1],
            "valid_to": [ts_2],
            "is_current": [False],
            "row_hash": ["hash_A"],
        },
        schema=schema,
    )

    # Snapshot at ts_3: ID=1 is back!
    snapshot = pl.DataFrame({"id": [1], "data": ["A"]})  # Data is same as old version, but it's a new life

    result = apply_scd2(snapshot, history, "id", ts_3, ["data"])

    # Expectation:
    # 1. Old record remains as is.
    # 2. New record created with valid_from=ts_3, valid_to=None, is_current=True.

    assert result.height == 2

    # Check old
    old = result.filter(pl.col("valid_from") == ts_1)
    assert old["valid_to"].item() == ts_2
    assert old["is_current"].item() is False

    # Check new
    new_rec = result.filter(pl.col("valid_from") == ts_3)
    assert new_rec.height == 1
    assert new_rec["valid_to"].item() is None
    assert new_rec["is_current"].item() is True


def test_scd2_null_hashing() -> None:
    """
    Edge Case: Hash columns contain Nulls.
    MD5 hashing usually fails on Null/None types if not handled.
    Code should fill nulls before hashing.
    """
    df = pl.DataFrame({"id": [1], "col_a": ["A"], "col_b": [None]})  # col_b is None

    # This should not raise an error
    hashed = generate_row_hash(df, ["col_a", "col_b"])

    assert "row_hash" in hashed.columns
    assert hashed["row_hash"].item() is not None
    # Ensure it's not empty string hash
    assert len(hashed["row_hash"].item()) == 32


def test_schema_refusal_missing_optional() -> None:
    """
    Edge Case: 'Refused' product with missing optional fields.
    Per FRD, therapeutic_area, atc_code etc. are Optional.
    """
    data: Dict[str, Any] = {
        "category": "Human",
        "product_number": "EMEA/H/C/9999",
        "medicine_name": "Refused Med",
        "marketing_authorisation_holder": "Bad Pharma",
        "active_substance": "Substance X",
        "authorisation_status": "Refused",
        "url": "http://example.com",
        # Missing therapeutic_area, atc_code
        # Missing business flags (should default)
    }

    row = EPARSourceRow(**data)
    assert row.product_number == "EMEA/H/C/9999"
    assert row.therapeutic_area is None
    assert row.atc_code is None
    assert row.biosimilar is False  # Default
    assert row.orphan is False  # Default


def test_schema_category_normalization() -> None:
    """
    Edge Case: Ingestion logic for Category normalization.
    (Note: This logic is in ingest.py, but we can verify the schema accepts 'Human' strictly)
    The schema is Literal['Human']. So 'human' should fail validation if passed directly.
    The ingestion layer is responsible for normalizing.
    """
    data: Dict[str, Any] = {
        "category": "human",  # Lowercase
        "product_number": "EMEA/H/C/001",
        "medicine_name": "M",
        "marketing_authorisation_holder": "H",
        "active_substance": "S",
        "authorisation_status": "A",
        "url": "u",
    }

    # Pydantic Literal is strict. 'human' != 'Human'.
    with pytest.raises(ValidationError):
        EPARSourceRow(**data)
