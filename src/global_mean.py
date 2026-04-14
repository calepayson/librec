import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

TARGET = "stars"


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _train_and_evaluate(name: str, output_path: Path) -> None:
    train = pd.read_parquet(DATA_DIR / f"{name}_train.parquet")
    val = pd.read_parquet(DATA_DIR / f"{name}_val.parquet")
    test = pd.read_parquet(DATA_DIR / f"{name}_test.parquet")

    logger.info(f"Computing global mean on {name}...")
    global_mean = float(train[TARGET].mean())

    val_rmse = _rmse(val[TARGET].to_numpy(), np.full(len(val), global_mean))
    test_rmse = _rmse(test[TARGET].to_numpy(), np.full(len(test), global_mean))

    lines = [
        f"=== {name} global-mean baseline ===",
        f"  Global mean: {global_mean:.4f}",
        f"  Val RMSE:    {val_rmse:.4f}",
        f"  Test RMSE:   {test_rmse:.4f}",
    ]
    output_path.write_text("\n".join(lines) + "\n")
    for line in lines:
        logger.info(line)


def lthing_global_mean(rebuild: bool = False) -> None:
    """Compute the global-mean baseline on LibraryThing and report RMSE."""
    logger.info("Getting lthing global-mean baseline results...")
    output_path = DATA_DIR / "lthing_global_mean.txt"

    if rebuild and output_path.exists():
        output_path.unlink()

    if output_path.exists():
        logger.info("lthing global-mean baseline already computed.")
        for line in output_path.read_text().splitlines():
            logger.info(line)
        return

    _train_and_evaluate("lthing", output_path)


def epinions_global_mean(rebuild: bool = False) -> None:
    """Compute the global-mean baseline on Epinions and report RMSE."""
    logger.info("Getting epinions global-mean baseline results...")
    output_path = DATA_DIR / "epinions_global_mean.txt"

    if rebuild and output_path.exists():
        output_path.unlink()

    if output_path.exists():
        logger.info("Epinions global-mean baseline already computed.")
        for line in output_path.read_text().splitlines():
            logger.info(line)
        return

    _train_and_evaluate("epinions", output_path)


def global_mean(rebuild: bool = False) -> None:
    """Compute global-mean baselines for all datasets."""
    lthing_global_mean(rebuild=rebuild)
    epinions_global_mean(rebuild=rebuild)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    global_mean()
