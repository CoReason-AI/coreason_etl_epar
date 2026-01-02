from typing import Dict
from unittest.mock import patch

import polars as pl
import pytest

from coreason_etl_epar.pipeline import EPARPipeline


@pytest.fixture
def hostile_env_postgres() -> Dict[str, str]:
    return {
        "PGUSER": "user@domain.com",
        "PGPASSWORD": "p@ssword!#$%^&*()",
        "PGHOST": "ipv6:::1",
        "PGPORT": "5432",
        "PGDATABASE": "db name with spaces",
    }


def test_load_bronze_postgres_credential_encoding(hostile_env_postgres: Dict[str, str]) -> None:
    """
    Verify that special characters in credentials are correctly URL-encoded.
    """
    with patch.dict("os.environ", hostile_env_postgres):
        with patch("polars.read_database") as mock_read_db:
            mock_read_db.return_value = pl.DataFrame()

            pipeline = EPARPipeline("dummy", "dummy", destination="postgres")

            with patch("dlt.pipeline"):
                pipeline.load_bronze()

            # Expected encoding:
            # user@domain.com -> user%40domain.com
            # p@ssword!#$%^&*() -> p%40ssword%21%23%24%25%5E%26%2A%28%29
            # ipv6:::1 -> ipv6:::1 (Host is usually not encoded in this simple f-string,
            # but user/pass MUST be).
            # Note: The implementation only applies quote_plus to user and password.

            # We check the connection arg of the call
            args, kwargs = mock_read_db.call_args_list[0]
            conn_str = kwargs["connection"]

            assert "user%40domain.com" in conn_str
            assert "p%40ssword%21%23%24%25%5E%26%2A%28%29" in conn_str
            # Ensure raw values are NOT present
            assert "user@domain.com" not in conn_str
            assert "p@ssword!" not in conn_str


def test_load_bronze_postgres_partial_failure() -> None:
    """
    Test scenario where EPAR fetch succeeds but SPOR fetch fails.
    Should return EPAR data and empty SPOR data.
    """
    mock_env = {"PGUSER": "u", "PGPASSWORD": "p", "PGHOST": "h", "PGPORT": "5432", "PGDATABASE": "d"}

    with patch.dict("os.environ", mock_env):
        with patch("polars.read_database") as mock_read_db:
            # EPAR succeeds, SPOR fails
            epar_df = pl.DataFrame({"id": [1]})
            mock_read_db.side_effect = [epar_df, Exception("SPOR Table Missing")]

            pipeline = EPARPipeline("dummy", "dummy", destination="postgres")

            with patch("dlt.pipeline"):
                result = pipeline.load_bronze()

            assert not result["epar"].is_empty()
            assert result["spor"].is_empty()
            assert result["epar"].shape == (1, 1)


def test_load_bronze_postgres_empty_result_sets() -> None:
    """
    Test scenario where queries return empty DataFrames (valid schema, 0 rows).
    This simulates "no data loaded yet" or "filtered out".
    """
    mock_env = {"PGUSER": "u", "PGPASSWORD": "p", "PGHOST": "h", "PGPORT": "5432", "PGDATABASE": "d"}

    with patch.dict("os.environ", mock_env):
        with patch("polars.read_database") as mock_read_db:
            # Return empty DFs with schema
            schema = {"col1": pl.Int64}
            empty_epar = pl.DataFrame(schema=schema)
            empty_spor = pl.DataFrame(schema=schema)

            mock_read_db.side_effect = [empty_epar, empty_spor]

            pipeline = EPARPipeline("dummy", "dummy", destination="postgres")

            with patch("dlt.pipeline"):
                result = pipeline.load_bronze()

            assert result["epar"].is_empty()
            assert "col1" in result["epar"].columns
            assert result["spor"].is_empty()
            assert "col1" in result["spor"].columns
