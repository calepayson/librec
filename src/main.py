import argparse
import logging

from download import download
from exploration import explore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--rebuild",
        nargs="?",
        const="all",
        metavar="STEP",
        help="Rebuild resources. Optionally specify a step: download, exploration (default: all)",
    )
    args = parser.parse_args()

    rebuild_download = args.rebuild in ("all", "download")
    rebuild_exploration = args.rebuild in ("all", "exploration")

    logger.info("Getting raw data...")
    download(rebuild=rebuild_download)

    logger.info("Starting basic data exploration...")
    explore(rebuild=rebuild_exploration)


if __name__ == "__main__":
    main()
