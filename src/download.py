import logging
import tarfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

DATASETS = {
    "lthing": "https://mcauleylab.ucsd.edu/public_datasets/data/librarything/lthing_data.tar.gz",
    "epinions": "https://mcauleylab.ucsd.edu/public_datasets/data/epinions/epinions_data.tar.gz",
}

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def _progress(block: int, block_size: int, total: int) -> None:
    """Print a progress bar to stdout for use as a urlretrieve reporthook.

    Args:
        block: Number of blocks transferred so far.
        block_size: Size of each block in bytes.
        total: Total size of the file in bytes.
    """
    downloaded = min(block * block_size, total)
    pct = downloaded / total * 100
    bar = "#" * (downloaded * 40 // total)
    mb = downloaded / 1_048_576
    total_mb = total / 1_048_576
    print(
        f"\r  [{bar:<40}] {pct:5.1f}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True
    )


def _download_and_extract(name: str, url: str) -> None:
    """Download a tar.gz archive from url and extract it into DATA_DIR.

    Creates DATA_DIR if it does not exist, downloads the archive with a
    progress bar, extracts its contents, then removes the archive file.

    Args:
        name: Short identifier for the dataset, used to name the archive file.
        url: URL of the tar.gz archive to download.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / f"{name}.tar.gz"

    logger.info(f"Downloading {name}...")
    urllib.request.urlretrieve(url, archive, reporthook=_progress)
    print()

    logger.info(f"Extracting {name}...")
    with tarfile.open(archive) as tf:
        tf.extractall(DATA_DIR)

    archive.unlink()


def download() -> None:
    """Download and extract all datasets listed in DATASETS.

    Skips any dataset whose extracted directory already exists in DATA_DIR.
    """
    for name, url in DATASETS.items():
        extracted = DATA_DIR / f"{name}_data"
        if extracted.exists():
            logger.info(f"Skipping {name} (already downloaded)")
            continue
        _download_and_extract(name, url)


if __name__ == "__main__":
    # Download all datasets when the module is run directly.
    download()
