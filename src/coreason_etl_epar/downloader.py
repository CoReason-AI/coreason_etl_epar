from pathlib import Path
from typing import Optional

import requests  # type: ignore[import-untyped]
from loguru import logger
from requests.adapters import HTTPAdapter  # type: ignore[import-untyped]
from urllib3.util.retry import Retry

# Endpoints defined in FRD
URL_EPAR_INDEX = (
    "https://www.ema.europa.eu/sites/default/files/Medicines_output_european_public_assessment_reports.xlsx"
)
URL_SPOR_EXPORT = "https://spor-net.ema.europa.eu/oms-api/v1/organisations/export"


def get_session(retries: int = 3, backoff_factor: float = 0.3) -> requests.Session:
    """
    Creates a requests Session with retry logic.
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(500, 502, 503, 504),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download_file(url: str, dest_path: Path, timeout: int = 60) -> None:
    """
    Downloads a file from a URL to a destination path using streaming.
    Uses atomic writing (download to .tmp then rename) and retries.

    Args:
        url: The URL to download from.
        dest_path: The local path to save the file.
        timeout: Request timeout in seconds.
    """
    logger.info(f"Downloading {url} to {dest_path}")

    # Ensure parent directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    session = get_session()

    try:
        # Stream=True to handle large files without loading into memory
        with session.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()

            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        # Atomic move
        temp_path.replace(dest_path)
        logger.info(f"Successfully downloaded {dest_path}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        # Clean up temp file if it exists
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:  # pragma: no cover
                pass
        raise
    finally:
        session.close()


def fetch_sources(output_dir: Path, epar_url: Optional[str] = None, spor_url: Optional[str] = None) -> None:
    """
    Downloads both EPAR and SPOR source files to the output directory.

    Args:
        output_dir: Directory where files will be saved.
        epar_url: Override for EPAR URL.
        spor_url: Override for SPOR URL.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    url_epar = epar_url or URL_EPAR_INDEX
    url_spor = spor_url or URL_SPOR_EXPORT

    # Filenames are derived from the pipeline expectations (though pipeline takes paths)
    # We'll use standard names
    epar_path = output_dir / "medicines_output_european_public_assessment_reports.xlsx"
    spor_path = output_dir / "organisations.zip"  # SPOR export is a zip

    logger.info("Starting source fetch...")

    try:
        download_file(url_epar, epar_path)
        download_file(url_spor, spor_path)
        logger.info("All sources fetched successfully.")
    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        raise
