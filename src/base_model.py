import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "stars"


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


class BaseModel(ABC):
    name: str

    @abstractmethod
    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None: ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray: ...

    def evaluate(
        self,
        dataset: str,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        rebuild: bool = False,
    ) -> None:
        output_path = OUTPUT_DIR / f"{dataset}_{self.name}.txt"

        if rebuild and output_path.exists():
            output_path.unlink()

        if output_path.exists():
            logger.info(f"{dataset} {self.name} already computed.")
            for line in output_path.read_text().splitlines():
                logger.info(line)
            return

        logger.info(f"Fitting {self.name} on {dataset}...")
        self.fit(train, val)

        val_pred = self.predict(val)
        test_pred = self.predict(test)

        val_rmse = _rmse(val[TARGET].to_numpy(), val_pred)
        test_rmse = _rmse(test[TARGET].to_numpy(), test_pred)

        lines = [
            f"=== {dataset} {self.name} ===",
            f"  Val RMSE:   {val_rmse:.4f}",
            f"  Test RMSE:  {test_rmse:.4f}",
        ]
        output_path.write_text("\n".join(lines) + "\n")
        for line in lines:
            logger.info(line)
