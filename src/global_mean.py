import logging

import numpy as np
import pandas as pd

from base_model import BaseModel

logger = logging.getLogger(__name__)

TARGET = "stars"


class GlobalMean(BaseModel):
    name = "global_mean"

    def __init__(self):
        self._mean = None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        self._mean = float(train[TARGET].mean())
        logger.info(f"  Global mean: {self._mean:.4f}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self._mean)
