import ast
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1


def _split_paths(name: str) -> dict[str, Path]:
    return {
        "train": OUTPUT_DIR / f"{name}_train.parquet",
        "val": OUTPUT_DIR / f"{name}_val.parquet",
        "test": OUTPUT_DIR / f"{name}_test.parquet",
    }


def _temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by time and split into train/val/test using TRAIN_FRAC / VAL_FRAC."""
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def _write_splits(
    name: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    paths = _split_paths(name)
    train.to_parquet(paths["train"], index=False)
    val.to_parquet(paths["val"], index=False)
    test.to_parquet(paths["test"], index=False)
    logger.info(f"  Train: {len(train):,}")
    logger.info(f"  Val:   {len(val):,}")
    logger.info(f"  Test:  {len(test):,}")


def lthing_split(rebuild: bool = False) -> None:
    """Parse LibraryThing reviews and write temporal train/val/test parquet files."""
    logger.info("Getting lthing split...")
    paths = _split_paths("lthing")

    if rebuild:
        for p in paths.values():
            p.unlink(missing_ok=True)

    if all(p.exists() for p in paths.values()):
        logger.info("lthing split already computed.")
        for kind, p in paths.items():
            logger.info(f"  {kind.capitalize():<6}{len(pd.read_parquet(p)):,}")
        return

    logger.info("No lthing split found. Computing... (~1.7M lines)")
    reviews_path = DATA_DIR / "lthing_data" / "reviews.txt"

    records = []
    with open(reviews_path) as f:
        for line in tqdm(f, desc="lthing reviews"):
            line = line.strip()
            if not line.startswith("reviews["):
                continue
            try:
                sep = line.index(" = ")
                records.append(ast.literal_eval(line[sep + 3 :]))
            except (ValueError, SyntaxError):
                continue

    df = pd.DataFrame(records)
    df = df.drop(columns=["time"]).rename(columns={"work": "item", "unixtime": "time"})
    df = df[["user", "item", "stars", "time"]].dropna()
    df["time"] = df["time"].astype("int64")

    train, val, test = _temporal_split(df)
    _write_splits("lthing", train, val, test)


def epinions_split(rebuild: bool = False) -> None:
    """Parse Epinions reviews and write temporal train/val/test parquet files."""
    logger.info("Getting epinions split...")
    paths = _split_paths("epinions")

    if rebuild:
        for p in paths.values():
            p.unlink(missing_ok=True)

    if all(p.exists() for p in paths.values()):
        logger.info("Epinions split already computed.")
        for kind, p in paths.items():
            logger.info(f"  {kind.capitalize():<6}{len(pd.read_parquet(p)):,}")
        return

    logger.info("No epinions split found. Computing... (~195K lines)")
    reviews_path = DATA_DIR / "epinions_data" / "epinions.txt"

    records = []
    with open(reviews_path, encoding="latin-1") as f:
        next(f)  # skip header
        for line in tqdm(f, desc="epinions reviews"):
            parts = line.strip().split(None, 5)
            if len(parts) < 5:
                continue
            try:
                records.append(
                    {
                        "item": parts[0],
                        "user": parts[1],
                        "time": int(parts[3]),
                        "stars": float(parts[4]),
                    }
                )
            except ValueError:
                continue

    df = pd.DataFrame(records)
    df = df[df["stars"].between(1, 5)]
    df = df[["user", "item", "stars", "time"]].dropna()

    train, val, test = _temporal_split(df)
    _write_splits("epinions", train, val, test)


def split(rebuild: bool = False) -> None:
    """Write train/val/test parquet files for all datasets."""
    lthing_split(rebuild=rebuild)
    epinions_split(rebuild=rebuild)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    split()
