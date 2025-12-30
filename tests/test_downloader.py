from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest
import requests
from requests.adapters import HTTPAdapter

from coreason_etl_epar.downloader import download_file, fetch_sources


@pytest.fixture
def mock_session_get() -> Iterator[MagicMock]:
    """
    Mocks requests.Session.get, since we now use a session with retries.
    """
    with patch("requests.Session.get") as mock_get:
        yield mock_get


def test_download_file_success(tmp_path: Path, mock_session_get: MagicMock) -> None:
    dest_path = tmp_path / "test_file.txt"
    content = b"some binary content"

    # Mock Response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content.return_value = [content]
    mock_session_get.return_value.__enter__.return_value = mock_response

    download_file("http://example.com/file", dest_path)

    assert dest_path.exists()
    assert dest_path.read_bytes() == content
    mock_session_get.assert_called_once()
    mock_response.raise_for_status.assert_called_once()


def test_download_file_http_error(tmp_path: Path, mock_session_get: MagicMock) -> None:
    dest_path = tmp_path / "fail.txt"

    # Mock Response raising HTTPError
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
    mock_session_get.return_value.__enter__.return_value = mock_response

    with pytest.raises(requests.exceptions.HTTPError):
        download_file("http://example.com/fail", dest_path)

    assert not dest_path.exists()


def test_download_file_atomic_write_failure(tmp_path: Path, mock_session_get: MagicMock) -> None:
    """
    Test that if download fails midway, no partial/final file is left.
    """
    dest_path = tmp_path / "atomic_fail.txt"

    # Mock Response that works initially but fails during streaming
    mock_response = MagicMock()
    mock_response.status_code = 200

    # iter_content yields one chunk then raises error
    def side_effect(*args: Any, **kwargs: Any) -> Iterator[bytes]:
        yield b"chunk1"
        raise requests.exceptions.ChunkedEncodingError("Stream broken")

    mock_response.iter_content.side_effect = side_effect
    mock_session_get.return_value.__enter__.return_value = mock_response

    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        download_file("http://example.com/atomic", dest_path)

    # Final file should not exist
    assert not dest_path.exists()
    # Temp file should be cleaned up
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    assert not temp_path.exists()


def test_fetch_sources_success(tmp_path: Path, mock_session_get: MagicMock) -> None:
    # Setup mock to return success for both calls
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content.return_value = [b"data"]
    mock_session_get.return_value.__enter__.return_value = mock_response

    fetch_sources(tmp_path)

    assert (tmp_path / "medicines_output_european_public_assessment_reports.xlsx").exists()
    assert (tmp_path / "organisations.zip").exists()

    # Should be called twice
    assert mock_session_get.call_count == 2


def test_fetch_sources_epar_failure(tmp_path: Path, mock_session_get: MagicMock) -> None:
    # Fail on EPAR (first call)
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("EPAR Error")
    mock_session_get.return_value.__enter__.return_value = mock_response

    with pytest.raises(requests.exceptions.HTTPError, match="EPAR Error"):
        fetch_sources(tmp_path)


def test_fetch_sources_spor_failure_no_cache(tmp_path: Path, mock_session_get: MagicMock) -> None:
    """
    Test that if SPOR download fails AND no cache exists, it raises exception.
    """

    # EPAR succeeds, SPOR fails
    def side_effect(url: str, **kwargs: Any) -> MagicMock:
        mock_resp = MagicMock()
        if "Medicines" in url:
            mock_resp.status_code = 200
            mock_resp.iter_content.return_value = [b"epar data"]
        else:
            mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("SPOR Error")
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    mock_session_get.side_effect = side_effect

    with pytest.raises(requests.exceptions.HTTPError, match="SPOR Error"):
        fetch_sources(tmp_path)


def test_fetch_sources_spor_resilience(tmp_path: Path, mock_session_get: MagicMock) -> None:
    """
    Test that if SPOR download fails BUT cache exists, it succeeds with warning.
    """
    # Setup Cache
    spor_cache = tmp_path / "organisations.zip"
    spor_cache.write_bytes(b"cached spor data")

    # EPAR succeeds, SPOR fails
    def side_effect(url: str, **kwargs: Any) -> MagicMock:
        mock_resp = MagicMock()
        if "Medicines" in url:
            mock_resp.status_code = 200
            mock_resp.iter_content.return_value = [b"epar data"]
        else:
            mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("SPOR Error")
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    mock_session_get.side_effect = side_effect

    fetch_sources(tmp_path)  # Should not raise

    assert (tmp_path / "medicines_output_european_public_assessment_reports.xlsx").exists()
    assert spor_cache.exists()
    assert spor_cache.read_bytes() == b"cached spor data"  # Still the old data


# Integration style test for Retry logic.
# Mocking at adapter level is hard, so we just mock Session to raise error first.
# However, requests.Session.get is what we mock. The retry logic happens inside the adapter which calls 'urlopen'.
# Given we are using standard `patch`, we can verify that we *configured* the retry.


def test_session_configuration() -> None:
    """
    Verifies that the session is configured with retries.
    """
    from coreason_etl_epar.downloader import get_session

    s = get_session(retries=5)
    adapter = s.get_adapter("https://")

    # BaseAdapter doesn't imply max_retries, but HTTPAdapter does.
    assert isinstance(adapter, HTTPAdapter)
    assert adapter.max_retries.total == 5
    s.close()
