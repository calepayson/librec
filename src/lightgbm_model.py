import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from base_model import BaseModel

logger = logging.getLogger(__name__)

TARGET = "stars"
CATEGORICAL = []

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
    "seed": 42,
}
LOG_EVERY = 10
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 100
OOF_FOLDS = 5
SEED = 42


def _safe_std(series: pd.Series) -> float:
    value = series.std()
    return 0.0 if pd.isna(value) else value


class _FeatureEngineer:
    def __init__(self):
        self.global_mean = 0.0
        self.user_stats = None
        self.item_stats = None
        self.user_item_counts = None
        self.feature_names = []

    def fit(self, train: pd.DataFrame) -> "_FeatureEngineer":
        self.global_mean = float(train[TARGET].mean())

        self.user_stats = self._stats(train, "user", "user")
        self.item_stats = self._stats(train, "item", "item")
        self.user_item_counts = (
            train.groupby(["user", "item"], observed=True)
            .size()
            .rename("user_item_train_count")
            .reset_index()
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        features = features.join(
            df[["user"]].merge(self.user_stats, on="user", how="left").drop(
                columns=["user"]
            )
        )
        features = features.join(
            df[["item"]].merge(self.item_stats, on="item", how="left").drop(
                columns=["item"]
            )
        )
        features = features.join(
            df[["user", "item"]]
            .merge(self.user_item_counts, on=["user", "item"], how="left")
            .drop(columns=["user", "item"])
        )
        features["user_item_train_count"] = features[
            "user_item_train_count"
        ].fillna(0)

        features["is_cold_user"] = features["user_count"].isna().astype("int8")
        features["is_cold_item"] = features["item_count"].isna().astype("int8")
        features["user_count"] = features["user_count"].fillna(0)
        features["item_count"] = features["item_count"].fillna(0)

        for prefix in ("user", "item"):
            features[f"{prefix}_mean"] = features[f"{prefix}_mean"].fillna(
                self.global_mean
            )
            features[f"{prefix}_raw_mean"] = features[f"{prefix}_mean"]
            features[f"{prefix}_median"] = features[f"{prefix}_median"].fillna(
                self.global_mean
            )
            features[f"{prefix}_std"] = features[f"{prefix}_std"].fillna(0)
            features[f"{prefix}_min"] = features[f"{prefix}_min"].fillna(
                self.global_mean
            )
            features[f"{prefix}_max"] = features[f"{prefix}_max"].fillna(
                self.global_mean
            )

            count = features[f"{prefix}_count"]
            mean = features[f"{prefix}_mean"]
            features[f"{prefix}_log_count"] = np.log1p(count)
            features[f"{prefix}_mean_minus_global"] = mean - self.global_mean
            features[f"{prefix}_bias"] = features[f"{prefix}_mean_minus_global"]
            features[f"{prefix}_rating_span"] = (
                features[f"{prefix}_max"] - features[f"{prefix}_min"]
            )

        features["user_item_popularity"] = (
            features["user_log_count"] * features["item_log_count"]
        )
        features["mean_user_item_rating"] = (
            features["user_mean"] + features["item_mean"]
        ) / 2
        features["user_item_bias_sum"] = features["user_bias"] + features["item_bias"]
        features["user_item_bias_gap"] = features["user_bias"] - features["item_bias"]
        features["item_minus_user_mean"] = features["item_mean"] - features["user_mean"]

        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        for col in CATEGORICAL:
            features[col] = features[col].astype("int32")

        numeric_cols = [c for c in features.columns if c not in CATEGORICAL]
        features[numeric_cols] = features[numeric_cols].astype("float32")
        return features

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        self.fit(train)
        features = self.transform(train)
        self.feature_names = list(features.columns)
        return features

    def fit_transform_oof(self, train: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(SEED)
        indices = np.arange(len(train))
        rng.shuffle(indices)

        fold_features = []
        for fold, valid_idx in enumerate(np.array_split(indices, OOF_FOLDS), start=1):
            train_idx = np.setdiff1d(indices, valid_idx, assume_unique=True)
            fold_engineer = _FeatureEngineer().fit(train.iloc[train_idx])
            features = fold_engineer.transform(train.iloc[valid_idx])
            features.index = train.index[valid_idx]
            fold_features.append(features)
            logger.info(
                "  Built OOF features for fold %s/%s (%s rows)",
                fold,
                OOF_FOLDS,
                len(valid_idx),
            )

        features = pd.concat(fold_features).sort_index()
        self.fit(train)
        self.feature_names = list(features.columns)
        return features

    @staticmethod
    def _stats(train: pd.DataFrame, group_col: str, prefix: str) -> pd.DataFrame:
        grouped = train.groupby(group_col, observed=True)
        stats = grouped.agg(
            count=(TARGET, "size"),
            mean=(TARGET, "mean"),
            median=(TARGET, "median"),
            std=(TARGET, _safe_std),
            min=(TARGET, "min"),
            max=(TARGET, "max"),
        ).reset_index()
        return stats.rename(
            columns={col: f"{prefix}_{col}" for col in stats.columns if col != group_col}
        )


class LightGBM(BaseModel):
    name = "lightgbm"

    def __init__(self):
        self._model = None
        self._features = _FeatureEngineer()
        self._predict_calls = 0

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        train_x = self._features.fit_transform_oof(train)
        val_x = self._features.transform(val)
        logger.info(f"  Engineered {len(train_x.columns)} LightGBM features")

        train_set = lgb.Dataset(
            train_x, label=train[TARGET], categorical_feature=CATEGORICAL
        )
        val_set = lgb.Dataset(
            val_x,
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
        self._log_feature_importance()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        features = self._features.transform(df)
        preds = self._model.predict(
            features, num_iteration=self._model.best_iteration
        )
        preds = np.clip(preds, 0.5, 5.0)
        self._log_prediction_distribution(preds)
        return preds

    def _log_prediction_distribution(self, preds: np.ndarray) -> None:
        self._predict_calls += 1
        split_name = "val" if self._predict_calls == 1 else "test"
        q1, median, q3 = np.quantile(preds, [0.25, 0.5, 0.75])
        logger.info(
            "  %s prediction distribution: mean=%.4f q1=%.4f median=%.4f q3=%.4f",
            split_name,
            float(np.mean(preds)),
            float(q1),
            float(median),
            float(q3),
        )

    def _log_feature_importance(self) -> None:
        importance = self._model.feature_importance(
            importance_type="gain",
            iteration=self._model.best_iteration,
        )
        total_gain = importance.sum()
        if total_gain <= 0:
            logger.info("  Feature importance unavailable: total gain is zero.")
            return

        feature_importance = sorted(
            zip(self._model.feature_name(), importance / total_gain * 100),
            key=lambda item: item[1],
            reverse=True,
        )
        logger.info("  Feature importance by gain:")
        for feature, pct in feature_importance:
            logger.info(f"    {feature}: {pct:.2f}%")
