import argparse
import itertools
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from base_model import _rmse
from lightgbm_model import (
    CATEGORICAL,
    EARLY_STOPPING_ROUNDS,
    LGB_PARAMS,
    LOG_EVERY,
    NUM_BOOST_ROUND,
    TARGET,
    _FeatureEngineer,
)
from preprocess import DATASETS, load_preprocessed, load_trust_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sweeps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARAM_GRID = {
    "feature_fraction": [0.75, 0.9, 1.0],
    "bagging_fraction": [0.75, 0.9, 1.0],
    "bagging_freq": [0, 5],
    "feature_fraction_bynode": [None, 0.8],
}


def _iter_param_grid() -> list[dict]:
    keys = list(PARAM_GRID)
    return [dict(zip(keys, values)) for values in itertools.product(*PARAM_GRID.values())]


def _make_params(overrides: dict) -> dict:
    overrides = {k: v for k, v in overrides.items() if v is not None}
    params = {
        **LGB_PARAMS,
        "learning_rate": 0.02,
        "num_leaves": 15,
        "min_data_in_leaf": 50,
        "lambda_l2": 0.0,
        **overrides,
        "verbosity": -1,
        "force_col_wise": True,
        "seed": 42,
    }
    return params


def sweep(dataset: str, limit: int | None = None) -> pd.DataFrame:
    logger.info("Loading preprocessed %s data...", dataset)
    train, val, _ = load_preprocessed(dataset)
    social_edges = load_trust_graph(dataset)

    if social_edges is None:
        logger.info("No social graph found for %s.", dataset)
    else:
        logger.info("Loaded social graph for %s (%s directed edges)", dataset, len(social_edges))

    logger.info("Building OOF train features once...")
    features = _FeatureEngineer(social_edges)
    train_x = features.fit_transform_oof(train)

    logger.info("Building validation features once...")
    val_x = features.transform(val)

    train_set = lgb.Dataset(
        train_x,
        label=train[TARGET],
        categorical_feature=CATEGORICAL,
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        val_x,
        label=val[TARGET],
        categorical_feature=CATEGORICAL,
        reference=train_set,
        free_raw_data=False,
    )
    val_true = val[TARGET].to_numpy()

    configs = _iter_param_grid()
    if limit is not None:
        configs = configs[:limit]

    logger.info("Sweeping %s LightGBM configurations...", len(configs))
    rows = []
    for i, overrides in enumerate(configs, start=1):
        params = _make_params(overrides)
        logger.info("Config %s/%s: %s", i, len(configs), overrides)
        model = lgb.train(
            params,
            train_set,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(LOG_EVERY),
            ],
        )
        val_pred = np.clip(
            model.predict(val_x, num_iteration=model.best_iteration), 0.5, 5.0
        )
        val_rmse = _rmse(val_true, val_pred)
        rows.append(
            {
                **overrides,
                "best_iteration": model.best_iteration,
                "val_rmse": val_rmse,
            }
        )
        logger.info("  val_rmse=%.6f best_iteration=%s", val_rmse, model.best_iteration)

    results = pd.DataFrame(rows).sort_values("val_rmse")
    out_path = OUTPUT_DIR / f"{dataset}_lightgbm_sweep.csv"
    results.to_csv(out_path, index=False)
    logger.info("Saved sweep results to %s", out_path)
    logger.info("Best config:\n%s", results.head(1).to_string(index=False))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        default="lthing",
        choices=DATASETS,
        help="Dataset to sweep (default: lthing)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only run the first N configs from the grid; useful for smoke tests.",
    )
    args = parser.parse_args()
    sweep(args.dataset, limit=args.limit)


if __name__ == "__main__":
    main()
