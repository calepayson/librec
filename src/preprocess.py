import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "preprocessed"
DATASETS = ["lthing", "epinions"]
TARGET = "stars"


def _encode(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add user_code/item_code columns using train-set vocabulary. Unseen -> -1."""
    logger.info("Encoding user/item IDs to integers...")
    user_cats = train["user"].astype("category").cat.categories
    item_cats = train["item"].astype("category").cat.categories
    user_to_code = {u: i for i, u in enumerate(user_cats)}
    item_to_code = {it: i for i, it in enumerate(item_cats)}
    logger.info(
        f"  {len(user_to_code):,} users, {len(item_to_code):,} items in vocabulary"
    )
    out = []
    for df in (train, val, test):
        df = df.copy()
        df["user_code"] = df["user"].map(user_to_code).fillna(-1).astype("int32")
        df["item_code"] = df["item"].map(item_to_code).fillna(-1).astype("int32")
        out.append(df)
    return out[0], out[1], out[2]


def _normalize(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add user_mean and stars_normalized columns. Unseen users get global mean."""
    logger.info("Normalizing ratings by user mean...")
    global_mean = train[TARGET].mean()
    user_means = train.groupby("user")[TARGET].mean()
    logger.info(f"  Global mean: {global_mean:.4f}")
    out = []
    for df in (train, val, test):
        df = df.copy()
        df["user_mean"] = (
            df["user"].map(user_means).fillna(global_mean).astype("float32")
        )
        df["stars_normalized"] = (df[TARGET] - df["user_mean"]).astype("float32")
        out.append(df)
    return out[0], out[1], out[2]


def _add_time_features(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add days_since_start using train min timestamp as reference."""
    logger.info("Adding time features...")
    min_time = train["time"].min()
    out = []
    for df in (train, val, test):
        df = df.copy()
        df["days_since_start"] = ((df["time"] - min_time) / 86400).astype("float32")
        out.append(df)
    return out[0], out[1], out[2]


def _preprocess_dataset(name: str, rebuild: bool) -> None:
    """Load split parquets, apply transforms, save preprocessed parquets."""
    splits = ("train", "val", "test")
    paths = [OUTPUT_DIR / split / f"{name}.parquet" for split in splits]

    if rebuild:
        for p in paths:
            if p.exists():
                p.unlink()

    if all(p.exists() for p in paths):
        logger.info(f"{name} preprocessed data already exists.")
        return

    for split in splits:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

    logger.info(f"Preprocessing {name}...")
    logger.info(f"Loading {name} split parquets...")
    train = pd.read_parquet(SPLITS_DIR / f"{name}_train.parquet")
    val = pd.read_parquet(SPLITS_DIR / f"{name}_val.parquet")
    test = pd.read_parquet(SPLITS_DIR / f"{name}_test.parquet")

    train, val, test = _encode(train, val, test)
    train, val, test = _normalize(train, val, test)
    train, val, test = _add_time_features(train, val, test)

    for df, path in zip((train, val, test), paths):
        df.to_parquet(path, index=False)
        logger.info(f"Saved {path} ({len(df):,} rows, {list(df.columns)})")


def preprocess(rebuild: bool = False) -> None:
    """Preprocess all datasets."""
    for name in DATASETS:
        _preprocess_dataset(name, rebuild)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocess()
