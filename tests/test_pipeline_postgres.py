from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from coreason_etl_epar.pipeline import EPARPipeline


@pytest.fixture
def mock_env_postgres():
    return {
        "PGUSER": "user",
        "PGPASSWORD": "password",
        "PGHOST": "localhost",
        "PGPORT": "5432",
        "PGDATABASE": "mydb",
    }


def test_load_bronze_postgres_success(mock_env_postgres):
    """
    Test successful loading from Postgres when env vars are present.
    """
    with patch.dict("os.environ", mock_env_postgres):
        with patch("polars.read_database") as mock_read_db:
            # Mock return values
            mock_read_db.side_effect = [
                pl.DataFrame({"col": [1]}),  # epar
                pl.DataFrame({"col": [2]}),  # spor
            ]

            pipeline = EPARPipeline("dummy", "dummy", destination="postgres")

            # Mock dlt pipeline to avoid real connection attempts
            with patch("dlt.pipeline") as mock_dlt:
                 # Ensure sql_client context manager works
                mock_dlt.return_value.sql_client.return_value.__enter__.return_value = MagicMock()

                result = pipeline.load_bronze()

            assert not result["epar"].is_empty()
            assert not result["spor"].is_empty()

            # Verify connection string construction
            expected_conn = "postgresql://user:password@localhost:5432/mydb"
            mock_read_db.assert_any_call(
                "SELECT * FROM bronze_epar.epar_index", connection=expected_conn
            )
            mock_read_db.assert_any_call(
                "SELECT * FROM bronze_epar.spor_organisations", connection=expected_conn
            )


def test_load_bronze_postgres_missing_env():
    """
    Test missing environment variables for Postgres.
    """
    # Ensure env is empty of PG vars
    with patch.dict("os.environ", {}, clear=True):
        pipeline = EPARPipeline("dummy", "dummy", destination="postgres")

        with patch("dlt.pipeline"):
            result = pipeline.load_bronze()

        # Should return empty dataframes and log error (checked via caplog if needed)
        assert result["epar"].is_empty()
        assert result["spor"].is_empty()


def test_load_bronze_postgres_connection_failure(mock_env_postgres):
    """
    Test failure during read_database (e.g. connection refused).
    """
    with patch.dict("os.environ", mock_env_postgres):
        with patch("polars.read_database") as mock_read_db:
            mock_read_db.side_effect = Exception("Connection Refused")

            pipeline = EPARPipeline("dummy", "dummy", destination="postgres")

            with patch("dlt.pipeline"):
                 result = pipeline.load_bronze()

            assert result["epar"].is_empty()
            assert result["spor"].is_empty()


def test_load_bronze_unsupported_destination():
    """
    Test fallback for unsupported destination.
    """
    pipeline = EPARPipeline("dummy", "dummy", destination="filesystem")

    with patch("dlt.pipeline"):
        result = pipeline.load_bronze()

    assert result["epar"].is_empty()
    assert result["spor"].is_empty()
