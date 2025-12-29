# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_epar

import sys
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from coreason_etl_epar.main import main


@pytest.fixture  # type: ignore[misc]
def mock_fetch_sources() -> Iterator[MagicMock]:
    with patch("coreason_etl_epar.main.fetch_sources") as mock:
        yield mock


@pytest.fixture  # type: ignore[misc]
def mock_pipeline() -> Iterator[MagicMock]:
    with patch("coreason_etl_epar.main.EPARPipeline") as mock:
        yield mock


def test_main_default_execution(mock_fetch_sources: MagicMock, mock_pipeline: MagicMock) -> None:
    """Test default execution: downloads files and runs pipeline."""
    with patch.object(sys, "argv", ["coreason-epar"]):
        main()

    # Verify fetch_sources called with default dir
    mock_fetch_sources.assert_called_once()
    args = mock_fetch_sources.call_args[0]
    assert str(args[0]) == ".coreason_data"

    # Verify pipeline execution
    mock_pipeline.assert_called_once()
    mock_pipeline.return_value.execute.assert_called_once()


def test_main_skip_download(mock_fetch_sources: MagicMock, mock_pipeline: MagicMock) -> None:
    """Test execution with --skip-download."""
    with patch.object(sys, "argv", ["coreason-epar", "--skip-download"]):
        main()

    mock_fetch_sources.assert_not_called()
    mock_pipeline.assert_called_once()
    mock_pipeline.return_value.execute.assert_called_once()


def test_main_custom_args(mock_fetch_sources: MagicMock, mock_pipeline: MagicMock) -> None:
    """Test execution with custom arguments."""
    with patch.object(sys, "argv", ["coreason-epar", "--work-dir", "custom_dir", "--destination", "postgres"]):
        main()

    # Verify fetch_sources called with custom dir
    mock_fetch_sources.assert_called_once()
    args = mock_fetch_sources.call_args[0]
    assert str(args[0]) == "custom_dir"

    # Verify pipeline initialized with custom destination
    mock_pipeline.assert_called_once()
    kwargs = mock_pipeline.call_args[1]
    assert kwargs["destination"] == "postgres"
    assert kwargs["work_dir"] == "custom_dir"


def test_main_exception_handling(mock_fetch_sources: MagicMock, mock_pipeline: MagicMock) -> None:
    """Test that exceptions cause a system exit with code 1."""
    mock_pipeline.return_value.execute.side_effect = RuntimeError("Pipeline crashed")

    with patch.object(sys, "argv", ["coreason-epar"]):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 1
