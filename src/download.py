import tarfile
import urllib.request
from pathlib import Path

DATASETS = {
    "lthing": "https://mcauleylab.ucsd.edu/public_datasets/data/librarything/lthing_data.tar.gz",
    "epinions": "https://mcauleylab.ucsd.edu/public_datasets/data/epinions/epinions_data.tar.gz",
}

DATA_DIR = Path(__file__).parent.parent / "data"


def _progress(block: int, block_size: int, total: int) -> None:
    downloaded = min(block * block_size, total)
    pct = downloaded / total * 100
    bar = "#" * (downloaded * 40 // total)
    mb = downloaded / 1_048_576
    total_mb = total / 1_048_576
    print(
        f"\r  [{bar:<40}] {pct:5.1f}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True
    )


def _download_and_extract(name: str, url: str) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    archive = DATA_DIR / f"{name}.tar.gz"

    print(f"Downloading {name}...")
    urllib.request.urlretrieve(url, archive, reporthook=_progress)
    print()

    print(f"Extracting {name}...")
    with tarfile.open(archive) as tf:
        tf.extractall(DATA_DIR)

    archive.unlink()


def download() -> None:
    for name, url in DATASETS.items():
        extracted = DATA_DIR / f"{name}_data"
        if extracted.exists():
            print(f"Skipping {name} (already downloaded)")
            continue
        _download_and_extract(name, url)
