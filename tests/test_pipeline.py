import pytest
from unittest.mock import patch, MagicMock
import polars as pl
import os
from coreason_etl_epar.pipeline import EPARPipeline

@pytest.fixture
def mock_dlt_pipeline():
    with patch("coreason_etl_epar.pipeline.dlt.pipeline") as mock:
        yield mock

def test_pipeline_init(tmp_path):
    p = EPARPipeline("epar.xlsx", "spor.zip", work_dir=str(tmp_path))
    assert p.epar_path == "epar.xlsx"
    assert p.work_dir.exists()

def test_run_ingestion_no_files(mock_dlt_pipeline, tmp_path):
    p = EPARPipeline("non_existent.xlsx", "non_existent.zip", work_dir=str(tmp_path))
    p.run_ingestion()
    # dlt.pipeline called
    mock_dlt_pipeline.assert_called()
    # run not called because files don't exist
    mock_dlt_pipeline.return_value.run.assert_not_called()

def test_run_ingestion_with_files(mock_dlt_pipeline, tmp_path):
    # Create dummy files
    epar = tmp_path / "epar.xlsx"
    epar.touch()
    spor = tmp_path / "spor.zip"
    spor.touch()

    p = EPARPipeline(str(epar), str(spor), work_dir=str(tmp_path))

    # Mock the resources to avoid actual file reading errors (since they are empty/dummy)
    with patch("coreason_etl_epar.pipeline.epar_index") as mock_epar, \
         patch("coreason_etl_epar.pipeline.spor_organisations") as mock_spor:

        p.run_ingestion()

        # Check run called twice
        assert mock_dlt_pipeline.return_value.run.call_count == 2

def test_load_bronze_empty(tmp_path):
    p = EPARPipeline("dummy", "dummy", work_dir=str(tmp_path))
    # No duckdb file exists
    data = p.load_bronze()
    assert data["epar"].is_empty()
    assert data["spor"].is_empty()

def test_execute_flow(tmp_path):
    # Integration style test mocking the internals
    p = EPARPipeline("dummy", "dummy", work_dir=str(tmp_path))

    with patch.object(p, 'run_ingestion') as mock_ingest, \
         patch.object(p, 'load_bronze') as mock_load, \
         patch.object(p, 'run_transformations') as mock_trans:

        # Case 1: No data
        mock_load.return_value = {"epar": pl.DataFrame(), "spor": pl.DataFrame()}
        p.execute()
        mock_ingest.assert_called_once()
        mock_trans.assert_not_called()

        # Case 2: Data present
        mock_ingest.reset_mock()
        mock_load.return_value = {"epar": pl.DataFrame({"a": [1]}), "spor": pl.DataFrame()}
        p.execute()
        mock_ingest.assert_called_once()
        mock_trans.assert_called_once()

def test_pipeline_real_execution(tmp_path):
    # Test internal logic of run_ingestion, load_bronze (partial), and run_transformations
    # by mocking only the I/O edge points but not the internal logic.

    p = EPARPipeline("epar.xlsx", "spor.zip", work_dir=str(tmp_path), destination="duckdb")

    # 1. Mock DLT pipeline to return a mock client that returns our dummy data
    with patch("coreason_etl_epar.pipeline.dlt.pipeline") as mock_dlt_class:
        # Configure the mock to support context manager for sql_client
        mock_pipeline_instance = mock_dlt_class.return_value
        mock_pipeline_instance.run.return_value = "Run Info"
        mock_pipeline_instance.sql_client.return_value.__enter__.return_value = MagicMock()

        # Mock Ingestion
        # Touch files
        (tmp_path / "epar.xlsx").touch()
        (tmp_path / "spor.zip").touch()

        p.run_ingestion()

        # 2. Test load_bronze internal logic
        # It constructs connection string and calls pl.read_database.

        # We need a more granular patch for os.path.exists

        original_exists = os.path.exists

        def side_effect_exists(path):
            if str(path).endswith(".duckdb"):
                return True
            return original_exists(path)

        with patch("os.path.exists", side_effect=side_effect_exists), \
             patch("polars.read_database") as mock_read_db:

            # Define what read_database returns
            # It's called twice (epar, spor)

            # Valid EPAR DF
            epar_df = pl.DataFrame({
                "category": ["Human"],
                "product_number": ["P1"],
                "medicine_name": ["M1"],
                "marketing_authorisation_holder": ["Holder A"],
                "active_substance": ["Sub A"],
                "atc_code": ["A01"],
                "authorisation_status": ["Authorised"],
                "url": ["u"],
                "biosimilar": [False],
                "generic": [False],
                "orphan": [False],
                "therapeutic_area": ["Area"]
            })

            # Valid SPOR DF
            spor_df = pl.DataFrame({
                "name": ["Holder A"],
                "org_id": ["ORG-1"],
                "roles": [["Marketing Authorisation Holder"]]
            })

            def side_effect(*args, **kwargs):
                query = args[0]
                if "epar_index" in query:
                    return epar_df
                if "spor_organisations" in query:
                    return spor_df
                return pl.DataFrame()

            mock_read_db.side_effect = side_effect

            data = p.load_bronze()

            assert not data["epar"].is_empty()

            # 3. Test run_transformations internal logic (Real run)
            # We pass the data we got
            # Note: run_transformations logic includes loading existing history.
            # Our exists mock delegates to original_exists, so silver_history.parquet check will behave correctly (false initially).

            p.run_transformations(data["epar"], data["spor"], history_path=str(tmp_path / "silver_history.parquet"))

            # Check if output files exist (use pathlib check which uses stat, but might be mocked? No, we mocked os.path.exists function only)
            # pathlib.Path.exists() calls os.stat usually.

            assert (tmp_path / "dim_medicine.parquet").exists()
            assert (tmp_path / "fact_regulatory_history.parquet").exists()
            assert (tmp_path / "silver_history.parquet").exists()

def test_load_bronze_duckdb_fail(tmp_path):
    p = EPARPipeline("dummy", "dummy", work_dir=str(tmp_path), destination="duckdb")

    # Mock exists=True but read_database fails
    with patch("os.path.exists", return_value=True), \
         patch("polars.read_database", side_effect=Exception("DB Error")):

        data = p.load_bronze()
        assert data["epar"].is_empty()
        assert data["spor"].is_empty()

def test_load_bronze_partial_fail(tmp_path):
    p = EPARPipeline("dummy", "dummy", work_dir=str(tmp_path), destination="duckdb")

    # Mock exists=True, but read_database fails ONLY for epar, succeeds for spor
    # Need to mock dlt pipeline client to succeed (return generic or configured mock) so it doesn't return empty early
    with patch("coreason_etl_epar.pipeline.dlt.pipeline") as mock_dlt_class, \
         patch("os.path.exists", return_value=True), \
         patch("polars.read_database") as mock_read:

        # Make the connection check pass
        mock_pipeline_instance = mock_dlt_class.return_value
        mock_pipeline_instance.sql_client.return_value.__enter__.return_value = MagicMock()

        def side_effect(query, **kwargs):
            if "epar_index" in query:
                raise Exception("EPAR Table Missing")
            if "spor_organisations" in query:
                return pl.DataFrame({"a": [1]})
            return pl.DataFrame()

        mock_read.side_effect = side_effect

        data = p.load_bronze()
        assert data["epar"].is_empty() # Failed
        assert not data["spor"].is_empty() # Succeeded

def test_load_bronze_spor_fail(tmp_path):
    p = EPARPipeline("dummy", "dummy", work_dir=str(tmp_path), destination="duckdb")

    with patch("coreason_etl_epar.pipeline.dlt.pipeline") as mock_dlt_class, \
         patch("os.path.exists", return_value=True), \
         patch("polars.read_database") as mock_read:

        # Make the connection check pass
        mock_pipeline_instance = mock_dlt_class.return_value
        mock_pipeline_instance.sql_client.return_value.__enter__.return_value = MagicMock()

        def side_effect(query, **kwargs):
            if "epar_index" in query:
                return pl.DataFrame({"a": [1]})
            if "spor_organisations" in query:
                raise Exception("SPOR Table Missing")
            return pl.DataFrame()

        mock_read.side_effect = side_effect

        data = p.load_bronze()
        assert not data["epar"].is_empty() # Succeeded
        assert data["spor"].is_empty() # Failed

def test_load_bronze_not_duckdb(tmp_path):
    # Test fallback when destination is not duckdb
    p = EPARPipeline("dummy", "dummy", work_dir=str(tmp_path), destination="postgres")

    # We must mock dlt pipeline to pass the initial connectivity check
    with patch("coreason_etl_epar.pipeline.dlt.pipeline") as mock_dlt_class:
        mock_pipeline_instance = mock_dlt_class.return_value
        # Ensure sql_client context manager works
        mock_pipeline_instance.sql_client.return_value.__enter__.return_value = MagicMock()

        data = p.load_bronze()
        assert data["epar"].is_empty()
        assert data["spor"].is_empty()

def test_load_bronze_duckdb_not_exists(tmp_path):
    # Destination is duckdb but file doesn't exist
    p = EPARPipeline("dummy", "dummy", work_dir=str(tmp_path), destination="duckdb")

    # We must mock dlt pipeline to pass the initial connectivity check
    with patch("coreason_etl_epar.pipeline.dlt.pipeline") as mock_dlt_class:
        mock_pipeline_instance = mock_dlt_class.return_value
        mock_pipeline_instance.sql_client.return_value.__enter__.return_value = MagicMock()

        # Don't mock exists, let it return False for file (which it will locally)
        data = p.load_bronze()
        assert data["epar"].is_empty()
