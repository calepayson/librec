import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from base_model import BaseModel

logger = logging.getLogger(__name__)

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
    "verbosity": -1,
    "force_col_wise": True,
}
LOG_EVERY = 10
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 20


class LightGBMBaseline(BaseModel):
    name = "baseline"

    def __init__(self):
        self._model = None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        train_set = lgb.Dataset(
            train[FEATURES], label=train[TARGET], categorical_feature=CATEGORICAL
        )
        val_set = lgb.Dataset(
            val[FEATURES],
            label=val[TARGET],
            categorical_feature=CATEGORICAL,
            reference=train_set,
        )

        self._model = lgb.train(
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
        logger.info(f"  Best iteration: {self._model.best_iteration}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._model.predict(
            df[FEATURES], num_iteration=self._model.best_iteration
        )
