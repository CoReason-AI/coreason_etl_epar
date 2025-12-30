from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from coreason_etl_epar.downloader import fetch_sources


@pytest.fixture
def mock_session_get_fail() -> Any:
    """
    Mock session that fails for SPOR but succeeds for EPAR.
    """
    with patch("requests.Session.get") as mock_get:

        def side_effect(url: str, **kwargs: Any) -> MagicMock:
            mock_resp = MagicMock()
            if "Medicines" in url:
                mock_resp.status_code = 200
                mock_resp.iter_content.return_value = [b"epar data"]
            else:
                mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("SPOR Error")
            mock_resp.__enter__.return_value = mock_resp
            return mock_resp

        mock_get.side_effect = side_effect
        yield mock_get


def test_spor_fallback_empty_file(tmp_path: Path, mock_session_get_fail: MagicMock) -> None:
    """
    Test that fallback fails if the cached file exists but is 0 bytes (empty).
    """
    spor_cache = tmp_path / "organisations.zip"
    spor_cache.touch()  # Create empty file (0 bytes)
    assert spor_cache.stat().st_size == 0

    with pytest.raises(requests.exceptions.HTTPError, match="SPOR Error"):
        fetch_sources(tmp_path)


def test_spor_fallback_directory(tmp_path: Path, mock_session_get_fail: MagicMock) -> None:
    """
    Test that fallback fails if the cached path exists but is a directory.
    """
    spor_cache = tmp_path / "organisations.zip"
    spor_cache.mkdir()  # Create directory instead of file
    assert spor_cache.exists()
    assert spor_cache.is_dir()

    with pytest.raises(requests.exceptions.HTTPError, match="SPOR Error"):
        fetch_sources(tmp_path)


def test_spor_fallback_valid_file(tmp_path: Path, mock_session_get_fail: MagicMock) -> None:
    """
    Test that fallback succeeds if the cached file is valid (> 0 bytes).
    """
    spor_cache = tmp_path / "organisations.zip"
    spor_cache.write_bytes(b"some data")
    assert spor_cache.stat().st_size > 0

    fetch_sources(tmp_path)  # Should not raise

    assert (tmp_path / "medicines_output_european_public_assessment_reports.xlsx").exists()
