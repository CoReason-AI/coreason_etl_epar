# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_epar

import argparse
import sys
from pathlib import Path

from coreason_etl_epar.downloader import fetch_sources
from coreason_etl_epar.logger import logger
from coreason_etl_epar.pipeline import EPARPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoReason EPAR ETL Pipeline")
    parser.add_argument(
        "--work-dir",
        type=str,
        default=".coreason_data",
        help="Directory for data storage (default: .coreason_data)",
    )
    parser.add_argument(
        "--destination",
        type=str,
        default="duckdb",
        help="DLT destination (default: duckdb)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading source files (use existing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)

    try:
        if not args.skip_download:
            logger.info(f"Downloading sources to {work_dir}...")
            fetch_sources(work_dir)
        else:
            logger.info("Skipping download step.")

        # Paths expected by pipeline (matching logic in downloader/pipeline)
        # Downloader saves as:
        # epar_path = output_dir / "medicines_output_european_public_assessment_reports.xlsx"
        # spor_path = output_dir / "organisations.zip"

        epar_path = work_dir / "medicines_output_european_public_assessment_reports.xlsx"
        spor_path = work_dir / "organisations.zip"

        logger.info(f"Initializing pipeline in {work_dir}...")
        pipeline = EPARPipeline(
            epar_path=str(epar_path),
            spor_path=str(spor_path),
            work_dir=str(work_dir),
            destination=args.destination,
        )

        pipeline.execute()
        logger.info("Pipeline finished successfully.")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()  # pragma: no cover
