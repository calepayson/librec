import argparse
import logging

from baseline import LightGBMBaseline
from download import download
from exploration import explore
from global_mean import GlobalMean
from preprocess import DATASETS, load_preprocessed, preprocess
from split import split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS = [
    GlobalMean(),
    LightGBMBaseline(),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--rebuild",
        nargs="?",
        const="all",
        metavar="STEP",
        help="Rebuild resources. Optionally specify a step: download, exploration, eda, split, preprocess, models (default: all)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="lthing",
        choices=DATASETS,
        help="Dataset to evaluate (default: lthing)",
    )
    args = parser.parse_args()

    rebuild_download = args.rebuild in ("all", "download")
    rebuild_exploration = args.rebuild in ("all", "exploration") or rebuild_download
    rebuild_eda = args.rebuild in ("all", "exploration", "eda") or rebuild_download
    rebuild_split = args.rebuild in ("all", "split") or rebuild_download
    rebuild_preprocess = args.rebuild in ("all", "preprocess") or rebuild_split
    rebuild_models = args.rebuild in ("all", "models") or rebuild_preprocess

    logger.info("Getting raw data...")
    download(args.dataset, rebuild=rebuild_download)

    logger.info("Starting exploration...")
    explore(args.dataset, rebuild=rebuild_exploration, rebuild_eda=rebuild_eda)

    logger.info("Building train/val/test splits...")
    split(args.dataset, rebuild=rebuild_split)

    logger.info("Preprocessing data...")
    preprocess(args.dataset, rebuild=rebuild_preprocess)

    logger.info("Running models...")
    all_results = []
    train, val, test = load_preprocessed(args.dataset)
    for model in MODELS:
        results = model.evaluate(args.dataset, train, val, test, rebuild=rebuild_models)
        all_results.append(results)


if __name__ == "__main__":
    main()
