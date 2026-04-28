import argparse
import logging
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from download import download
from exploration import explore
from global_mean import GlobalMean
from plot import plot
from preprocess import DATASETS, load_preprocessed, preprocess
from split import split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MODEL_NAMES = ["global_mean", "baseline", "gfp", "lightgbm", "ncf", "social_ncf"]


def _make_model(name):
    if name == "global_mean":
        return GlobalMean()
    if name == "baseline":
        from baseline import LightGBMBaseline

        return LightGBMBaseline()
    if name == "gfp":
        from gfp import GFP

        return GFP()
    if name == "lightgbm":
        from lightgbm_model import LightGBM

        return LightGBM()
    if name == "ncf":
        import torch
        from ncf import NCF

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        return NCF()
    if name == "social_ncf":
        import torch
        from social_ncf import SocialNCF

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        return SocialNCF()
    raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--rebuild",
        nargs="?",
        const="all",
        metavar="STEP",
        help="Rebuild resources. Optionally specify a step: download, exploration, eda, split, preprocess, models, plots (default: all)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="lthing",
        choices=DATASETS,
        help="Dataset to evaluate (default: lthing)",
    )
    parser.add_argument(
        "--only-model",
        choices=MODEL_NAMES,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    rebuild_download = args.rebuild in ("all", "download")
    rebuild_exploration = args.rebuild in ("all", "exploration") or rebuild_download
    rebuild_eda = args.rebuild in ("all", "exploration", "eda") or rebuild_download
    rebuild_split = args.rebuild in ("all", "split") or rebuild_download
    rebuild_preprocess = args.rebuild in ("all", "preprocess") or rebuild_split
    rebuild_models = args.rebuild in ("all", "models") or rebuild_preprocess
    rebuild_plots = args.rebuild in ("all", "plots") or rebuild_models
    model_names_to_rebuild = (
        {args.rebuild}
        if args.rebuild
        not in (
            None,
            "all",
            "download",
            "exploration",
            "eda",
            "split",
            "preprocess",
            "models",
            "plots",
        )
        else set()
    )

    if args.only_model:
        train, val, test = load_preprocessed(args.dataset)
        model = _make_model(args.only_model)
        rebuild = rebuild_models or model.name in model_names_to_rebuild
        return [model.evaluate(args.dataset, train, val, test, rebuild=rebuild)]

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
    model_names = (
        [name for name in MODEL_NAMES if name in model_names_to_rebuild]
        if model_names_to_rebuild
        else MODEL_NAMES
    )
    for model_name in model_names:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--dataset",
            args.dataset,
            "--only-model",
            model_name,
        ]
        if rebuild_models:
            command.extend(["--rebuild", "models"])
        elif model_name in model_names_to_rebuild:
            command.extend(["--rebuild", model_name])

        subprocess.run(command, check=True)

    logger.info("Generating plots...")
    plot(args.dataset, rebuild=rebuild_plots)

    return all_results


if __name__ == "__main__":
    main()
