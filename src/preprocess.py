import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "preprocessed"
DATASETS = ["lthing", "epinions"]
TARGET = "stars"

TRUST_FILES = {
    "lthing": RAW_DIR / "lthing_data" / "edges.txt",
}


def _encode(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Add user_code/item_code columns.

    Train users get codes 0..n_train-1. Val/test-only users get codes
    n_train..n_all-1 so they can be linked via the trust graph even though
    they have no trained embeddings. Items remain train-only (unseen -> -1).
    """
    logger.info("Encoding user/item IDs to integers...")
    train_user_cats = train["user"].astype("category").cat.categories
    user_to_code = {u: i for i, u in enumerate(train_user_cats)}

    next_code = len(train_user_cats)
    for u in pd.concat([val["user"], test["user"]]).unique():
        if u not in user_to_code:
            user_to_code[u] = next_code
            next_code += 1

    item_cats = train["item"].astype("category").cat.categories
    item_to_code = {it: i for i, it in enumerate(item_cats)}
    logger.info(
        f"  {len(train_user_cats):,} train users, {next_code:,} total users, "
        f"{len(item_to_code):,} items in vocabulary"
    )
    out = []
    for df in (train, val, test):
        df = df.copy()
        df["user_code"] = df["user"].map(user_to_code).fillna(-1).astype("int32")
        df["item_code"] = df["item"].map(item_to_code).fillna(-1).astype("int32")
        out.append(df)
    return out[0], out[1], out[2], user_to_code


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


def _encode_trust_graph(name: str, user_to_code: dict) -> None:
    """Load raw social/trust edges, encode to user_codes, and save as parquet."""
    raw_path = TRUST_FILES.get(name)
    if raw_path is None or not raw_path.exists():
        logger.info(f"No trust graph found for {name}, skipping.")
        return

    logger.info(f"Encoding trust graph for {name}...")
    edges = []
    with open(raw_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                src = user_to_code.get(parts[0])
                dst = user_to_code.get(parts[1])
                if src is not None and dst is not None:
                    edges.append((src, dst))
                    edges.append((dst, src))

    df = pd.DataFrame(edges, columns=["src", "dst"]).drop_duplicates()
    out_path = OUTPUT_DIR / f"{name}_trust.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(
        f"  Saved {out_path} ({len(df):,} directed edges, "
        f"{df['src'].nunique():,} users with neighbors)"
    )


def load_trust_graph(name: str) -> pd.DataFrame | None:
    """Load encoded trust graph edges, or None if unavailable."""
    path = OUTPUT_DIR / f"{name}_trust.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _preprocess_dataset(name: str, rebuild: bool) -> None:
    """Load split parquets, apply transforms, save preprocessed parquets."""
    splits = ("train", "val", "test")
    paths = [OUTPUT_DIR / split / f"{name}.parquet" for split in splits]

    trust_path = OUTPUT_DIR / f"{name}_trust.parquet"

    if rebuild:
        for p in paths:
            if p.exists():
                p.unlink()
        trust_path.unlink(missing_ok=True)

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

    train, val, test, user_to_code = _encode(train, val, test)
    _encode_trust_graph(name, user_to_code)
    train, val, test = _normalize(train, val, test)
    train, val, test = _add_time_features(train, val, test)

    for df, path in zip((train, val, test), paths):
        df.to_parquet(path, index=False)
        logger.info(f"Saved {path} ({len(df):,} rows, {list(df.columns)})")


def load_preprocessed(name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed train/val/test DataFrames for a dataset."""
    train = pd.read_parquet(OUTPUT_DIR / "train" / f"{name}.parquet")
    val = pd.read_parquet(OUTPUT_DIR / "val" / f"{name}.parquet")
    test = pd.read_parquet(OUTPUT_DIR / "test" / f"{name}.parquet")
    return train, val, test


def preprocess(dataset: str, rebuild: bool = False) -> None:
    """Preprocess the specified dataset."""
    _preprocess_dataset(dataset, rebuild)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocess()
