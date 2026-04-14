import argparse
import logging

from baseline import baseline
from download import download
from exploration import explore
from split import split

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
        help="Rebuild resources. Optionally specify a step: download, exploration, split, baseline (default: all)",
    )
    args = parser.parse_args()

    rebuild_download = args.rebuild in ("all", "download")
    rebuild_exploration = args.rebuild in ("all", "exploration")
    rebuild_split = args.rebuild in ("all", "split")
    rebuild_baseline = args.rebuild in ("all", "baseline")

    logger.info("Getting raw data...")
    download(rebuild=rebuild_download)

    logger.info("Starting basic data exploration...")
    explore(rebuild=rebuild_exploration)

    logger.info("Building train/val/test splits...")
    split(rebuild=rebuild_split)

    logger.info("Training LightGBM baselines...")
    baseline(rebuild=rebuild_baseline)


if __name__ == "__main__":
    main()
