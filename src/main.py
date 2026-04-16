import argparse
import logging

from baseline import baseline
from download import download
from exploration import explore
from global_mean import global_mean
from preprocess import preprocess
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
        help="Rebuild resources. Optionally specify a step: download, exploration, eda, split, preprocess, global_mean, baseline (default: all)",
    )
    args = parser.parse_args()

    rebuild_download = args.rebuild in ("all", "download")
    rebuild_exploration = args.rebuild in ("all", "exploration")
    rebuild_eda = args.rebuild in ("all", "exploration", "eda")
    rebuild_split = args.rebuild in ("all", "split")
    rebuild_preprocess = args.rebuild in ("all", "preprocess")
    rebuild_global_mean = args.rebuild in ("all", "global_mean")
    rebuild_baseline = args.rebuild in ("all", "baseline")

    logger.info("Getting raw data...")
    download(rebuild=rebuild_download)

    logger.info("Starting exploration...")
    explore(rebuild=rebuild_exploration, rebuild_eda=rebuild_eda)

    logger.info("Building train/val/test splits...")
    split(rebuild=rebuild_split)

    logger.info("Preprocessing data...")
    preprocess(rebuild=rebuild_preprocess)

    logger.info("Computing global-mean baselines...")
    global_mean(rebuild=rebuild_global_mean)

    logger.info("Training LightGBM baselines...")
    baseline(rebuild=rebuild_baseline)


if __name__ == "__main__":
    main()
