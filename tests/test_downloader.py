from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from coreason_etl_epar.downloader import URL_EPAR_INDEX, URL_SPOR_EXPORT, download_file, fetch_sources


@pytest.fixture
def mock_requests_get():
    with patch("requests.get") as mock_get:
        yield mock_get


def test_download_file_success(tmp_path: Path, mock_requests_get: MagicMock) -> None:
    dest_path = tmp_path / "test_file.txt"
    content = b"some binary content"

    # Mock Response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content.return_value = [content]
    mock_requests_get.return_value.__enter__.return_value = mock_response

    download_file("http://example.com/file", dest_path)

    assert dest_path.exists()
    assert dest_path.read_bytes() == content
    mock_requests_get.assert_called_once()
    mock_response.raise_for_status.assert_called_once()


def test_download_file_http_error(tmp_path: Path, mock_requests_get: MagicMock) -> None:
    dest_path = tmp_path / "fail.txt"

    # Mock Response raising HTTPError
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
    mock_requests_get.return_value.__enter__.return_value = mock_response

    with pytest.raises(requests.exceptions.HTTPError):
        download_file("http://example.com/fail", dest_path)

    assert not dest_path.exists()


def test_download_file_connection_error(tmp_path: Path, mock_requests_get: MagicMock) -> None:
    dest_path = tmp_path / "fail_conn.txt"

    # Mock Request raising ConnectionError immediately
    mock_requests_get.side_effect = requests.exceptions.ConnectionError("Connection Refused")

    with pytest.raises(requests.exceptions.ConnectionError):
        download_file("http://example.com/fail", dest_path)


def test_fetch_sources_success(tmp_path: Path, mock_requests_get: MagicMock) -> None:
    # Setup mock to return success for both calls
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content.return_value = [b"data"]
    mock_requests_get.return_value.__enter__.return_value = mock_response

    fetch_sources(tmp_path)

    assert (tmp_path / "medicines_output_european_public_assessment_reports.xlsx").exists()
    assert (tmp_path / "organisations.zip").exists()

    # Should be called twice
    assert mock_requests_get.call_count == 2

    # Verify calls
    calls = mock_requests_get.call_args_list
    assert calls[0][0][0] == URL_EPAR_INDEX
    assert calls[1][0][0] == URL_SPOR_EXPORT


def test_fetch_sources_failure(tmp_path: Path, mock_requests_get: MagicMock) -> None:
    # Fail on first call
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Error")
    mock_requests_get.return_value.__enter__.return_value = mock_response

    with pytest.raises(requests.exceptions.HTTPError):
        fetch_sources(tmp_path)
