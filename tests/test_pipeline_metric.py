import logging
from datetime import datetime
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import polars as pl
import pytest
from coreason_etl_epar.pipeline import EPARPipeline
from loguru import logger


@pytest.fixture  # type: ignore[misc]
def caplog(caplog: pytest.LogCaptureFixture) -> Generator[pytest.LogCaptureFixture, None, None]:
    class PropagateHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logging.getLogger(record.name).handle(record)

    handler_id = logger.add(PropagateHandler(), format="{message}")
    yield caplog
    logger.remove(handler_id)


def test_pipeline_scd_metric_logging(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # Verify that scd_updates_count is logged
    p = EPARPipeline("dummy", "dummy", work_dir=str(tmp_path))

    epar_df = pl.DataFrame(
        {
            "product_number": ["P1"],
            "medicine_name": ["M1"],
            "marketing_authorisation_holder": ["H1"],
            "authorisation_status": ["A"],
            "active_substance": ["S1"],
            "atc_code": ["A1"],
            "url": ["u"],
            "category": ["Human"],
        }
    )
    spor_df = pl.DataFrame({"name": ["H1"], "org_id": ["O1"]})

    with (
        patch("coreason_etl_epar.pipeline.apply_scd2") as mock_scd2,
        patch("coreason_etl_epar.pipeline.enrich_epar") as mock_enrich,
        patch("coreason_etl_epar.pipeline.create_gold_layer") as mock_gold,
    ):
        # Setup mock return for scd2
        mock_scd2.return_value = pl.DataFrame(
            {"valid_from": [datetime(2024, 1, 1)]}  # Arbitrary date
        )
        mock_enrich.return_value = pl.DataFrame()
        mock_gold.return_value = {}

        # Run with patched datetime to match our mock data
        with patch("coreason_etl_epar.pipeline.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 1)
            p.run_transformations(epar_df, spor_df)

        # Check logs
        assert "SCD Updates Count: 1" in caplog.text
