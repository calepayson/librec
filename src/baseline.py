import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "preprocessed"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["user_code", "item_code"]
CATEGORICAL = ["user_code", "item_code"]
TARGET = "stars"

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbosity": 1,
}
LOG_EVERY = 10
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 20


def _encode(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Encode user/item as integer codes using train categories; unseen -> -1."""
    out = []
    user_cats = train["user"].astype("category").cat.categories
    item_cats = train["item"].astype("category").cat.categories
    for df in (train, val, test):
        encoded = pd.DataFrame(
            {
                "user_code": pd.Categorical(
                    df["user"], categories=user_cats
                ).codes.astype("int32"),
                "item_code": pd.Categorical(
                    df["item"], categories=item_cats
                ).codes.astype("int32"),
                TARGET: df[TARGET].to_numpy(),
            }
        )
        out.append(encoded)
    return out[0], out[1], out[2]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _train_and_evaluate(name: str, output_path: Path) -> None:
    train = pd.read_parquet(DATA_DIR / "train" / f"{name}.parquet")
    val = pd.read_parquet(DATA_DIR / "val" / f"{name}.parquet")
    test = pd.read_parquet(DATA_DIR / "test" / f"{name}.parquet")

    train_enc, val_enc, test_enc = _encode(train, val, test)

    train_set = lgb.Dataset(
        train_enc[FEATURES], label=train_enc[TARGET], categorical_feature=CATEGORICAL
    )
    val_set = lgb.Dataset(
        val_enc[FEATURES],
        label=val_enc[TARGET],
        categorical_feature=CATEGORICAL,
        reference=train_set,
    )

    logger.info(f"Training LightGBM on {name}...")
    model = lgb.train(
        LGB_PARAMS,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(LOG_EVERY),
        ],
    )

    val_pred = model.predict(val_enc[FEATURES], num_iteration=model.best_iteration)
    test_pred = model.predict(test_enc[FEATURES], num_iteration=model.best_iteration)

    val_rmse = _rmse(val_enc[TARGET].to_numpy(), val_pred)
    test_rmse = _rmse(test_enc[TARGET].to_numpy(), test_pred)

    lines = [
        f"=== {name} LightGBM baseline ===",
        f"  Best iter:  {model.best_iteration}",
        f"  Val RMSE:   {val_rmse:.4f}",
        f"  Test RMSE:  {test_rmse:.4f}",
    ]
    output_path.write_text("\n".join(lines) + "\n")
    for line in lines:
        logger.info(line)


def lthing_baseline(rebuild: bool = False) -> None:
    """Train a LightGBM baseline on LibraryThing and report RMSE."""
    logger.info("Getting lthing baseline results...")
    output_path = OUTPUT_DIR / "lthing_baseline.txt"

    if rebuild and output_path.exists():
        output_path.unlink()

    if output_path.exists():
        logger.info("lthing baseline already computed.")
        for line in output_path.read_text().splitlines():
            logger.info(line)
        return

    _train_and_evaluate("lthing", output_path)


def epinions_baseline(rebuild: bool = False) -> None:
    """Train a LightGBM baseline on Epinions and report RMSE."""
    logger.info("Getting epinions baseline results...")
    output_path = OUTPUT_DIR / "epinions_baseline.txt"

    if rebuild and output_path.exists():
        output_path.unlink()

    if output_path.exists():
        logger.info("Epinions baseline already computed.")
        for line in output_path.read_text().splitlines():
            logger.info(line)
        return

    _train_and_evaluate("epinions", output_path)


def baseline(rebuild: bool = False) -> None:
    """Train and evaluate LightGBM baselines for all datasets."""
    lthing_baseline(rebuild=rebuild)
    epinions_baseline(rebuild=rebuild)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    baseline()
