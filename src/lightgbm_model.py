import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from base_model import BaseModel
from preprocess import load_trust_graph

logger = logging.getLogger(__name__)

TARGET = "stars"
CATEGORICAL = []

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.02,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
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
SOCIAL_FEATURES = [
    "social_degree",
    "social_log_degree",
    "social_neighbor_count",
    "social_neighbor_coverage",
    "social_neighbor_rating_count",
    "social_neighbor_log_rating_count",
    "social_neighbor_mean",
    "social_neighbor_mean_minus_global",
    "social_neighbor_mean_std",
    "social_neighbor_mean_min",
    "social_neighbor_mean_max",
    "social_neighbor_mean_range",
    "social_neighbor_weighted_mean",
    "social_neighbor_weighted_mean_minus_global",
    "social_user_vs_neighbor_mean",
    "social_item_vs_neighbor_mean",
    "social_has_neighbors",
    "social_has_rated_neighbors",
]
SOCIAL_ITEM_FEATURES = [
    "social_item_neighbor_count",
    "social_item_neighbor_log_count",
    "social_item_neighbor_mean",
    "social_item_neighbor_mean_minus_global",
    "social_item_neighbor_std",
    "social_item_neighbor_min",
    "social_item_neighbor_max",
    "social_item_neighbor_range",
    "social_item_neighbor_frac_liked",
    "social_item_neighbor_frac_disliked",
    "social_item_has_neighbor_rating",
    "social_item_user_gap",
    "social_item_item_gap",
    "social_item_neighbor_share",
]


def _safe_std(series: pd.Series) -> float:
    value = series.std()
    return 0.0 if pd.isna(value) else value


class _FeatureEngineer:
    def __init__(self, social_edges: pd.DataFrame | None = None):
        self.global_mean = 0.0
        self.social_edges = social_edges
        self.user_stats = None
        self.item_stats = None
        self.user_item_counts = None
        self.social_stats = None
        self.neighbor_item_ratings = None
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
        self.social_stats = self._social_stats(train)
        self.neighbor_item_ratings = self._neighbor_item_ratings(train)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        features = features.join(self._merge_on_index(df, ["user"], self.user_stats))
        features = features.join(self._merge_on_index(df, ["item"], self.item_stats))
        features = features.join(
            self._merge_on_index(df, ["user", "item"], self.user_item_counts)
        )
        features = features.join(
            self._merge_on_index(df, ["user_code"], self.social_stats)
        )
        features = features.join(self._social_item_features(df))
        features["user_item_train_count"] = features[
            "user_item_train_count"
        ].fillna(0)

        features["is_cold_user"] = features["user_count"].isna().astype("int8")
        features["is_cold_item"] = features["item_count"].isna().astype("int8")
        features["user_count"] = features["user_count"].fillna(0)
        features["item_count"] = features["item_count"].fillna(0)
        features["social_degree"] = features["social_degree"].fillna(0)
        features["social_neighbor_count"] = features["social_neighbor_count"].fillna(0)
        features["social_neighbor_rating_count"] = features[
            "social_neighbor_rating_count"
        ].fillna(0)

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

        features = self._fill_social_features(features)
        features = self._fill_social_item_features(features)

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
            fold_engineer = _FeatureEngineer(self.social_edges).fit(
                train.iloc[train_idx]
            )
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
    def _merge_on_index(
        df: pd.DataFrame, columns: list[str], stats: pd.DataFrame
    ) -> pd.DataFrame:
        values = df[columns].reset_index().merge(stats, on=columns, how="left")
        values = values.set_index("index").drop(columns=columns)
        return values.reindex(df.index)

    def _social_stats(self, train: pd.DataFrame) -> pd.DataFrame:
        if self.social_edges is None or self.social_edges.empty:
            return pd.DataFrame(columns=["user_code", *SOCIAL_FEATURES])

        user_stats = (
            train.groupby("user_code", observed=True)[TARGET]
            .agg(
                neighbor_rating_count="size",
                neighbor_mean="mean",
                neighbor_std=_safe_std,
                neighbor_min="min",
                neighbor_max="max",
            )
            .reset_index()
            .rename(columns={"user_code": "dst"})
        )

        edges = self.social_edges[["src", "dst"]].drop_duplicates()
        social = edges.merge(user_stats, on="dst", how="left")
        social["neighbor_weighted_sum"] = (
            social["neighbor_mean"] * social["neighbor_rating_count"]
        )

        grouped = social.groupby("src", observed=True)
        stats = grouped.agg(
            social_degree=("dst", "size"),
            social_neighbor_count=("neighbor_rating_count", "count"),
            social_neighbor_rating_count=("neighbor_rating_count", "sum"),
            social_neighbor_mean=("neighbor_mean", "mean"),
            social_neighbor_mean_std=("neighbor_mean", _safe_std),
            social_neighbor_mean_min=("neighbor_mean", "min"),
            social_neighbor_mean_max=("neighbor_mean", "max"),
            social_neighbor_weighted_sum=("neighbor_weighted_sum", "sum"),
        ).reset_index()

        stats["social_neighbor_weighted_mean"] = (
            stats["social_neighbor_weighted_sum"]
            / stats["social_neighbor_rating_count"].replace(0, np.nan)
        )
        stats = stats.drop(columns=["social_neighbor_weighted_sum"])
        return stats.rename(columns={"src": "user_code"})

    @staticmethod
    def _neighbor_item_ratings(train: pd.DataFrame) -> pd.DataFrame:
        ratings = train[["user_code", "item", TARGET]].copy()
        ratings["neighbor_item_liked"] = (ratings[TARGET] >= 4.0).astype("float32")
        ratings["neighbor_item_disliked"] = (ratings[TARGET] <= 2.0).astype("float32")
        ratings = (
            ratings.groupby(["user_code", "item"], observed=True)
            .agg(
                neighbor_item_rating=(TARGET, "mean"),
                neighbor_item_liked=("neighbor_item_liked", "mean"),
                neighbor_item_disliked=("neighbor_item_disliked", "mean"),
            )
            .reset_index()
            .rename(columns={"user_code": "dst"})
        )
        return ratings

    def _social_item_features(self, df: pd.DataFrame) -> pd.DataFrame:
        empty = pd.DataFrame(index=df.index, columns=SOCIAL_ITEM_FEATURES)
        if (
            self.social_edges is None
            or self.social_edges.empty
            or self.neighbor_item_ratings is None
            or self.neighbor_item_ratings.empty
        ):
            return empty

        row_items = (
            df[["user_code", "item"]]
            .reset_index()
            .rename(columns={"index": "_row_index", "user_code": "src"})
        )
        edges = self.social_edges[["src", "dst"]].drop_duplicates()
        matches = row_items.merge(edges, on="src", how="inner").merge(
            self.neighbor_item_ratings, on=["dst", "item"], how="inner"
        )
        if matches.empty:
            return empty

        grouped = matches.groupby("_row_index", observed=True)
        stats = grouped.agg(
            social_item_neighbor_count=("neighbor_item_rating", "size"),
            social_item_neighbor_mean=("neighbor_item_rating", "mean"),
            social_item_neighbor_std=("neighbor_item_rating", _safe_std),
            social_item_neighbor_min=("neighbor_item_rating", "min"),
            social_item_neighbor_max=("neighbor_item_rating", "max"),
            social_item_neighbor_frac_liked=("neighbor_item_liked", "mean"),
            social_item_neighbor_frac_disliked=("neighbor_item_disliked", "mean"),
        )
        return stats.reindex(df.index)

    def _fill_social_features(self, features: pd.DataFrame) -> pd.DataFrame:
        features["social_log_degree"] = np.log1p(features["social_degree"])
        features["social_neighbor_log_rating_count"] = np.log1p(
            features["social_neighbor_rating_count"]
        )
        features["social_neighbor_coverage"] = (
            features["social_neighbor_count"]
            / features["social_degree"].replace(0, np.nan)
        )
        features["social_has_neighbors"] = (features["social_degree"] > 0).astype(
            "int8"
        )
        features["social_has_rated_neighbors"] = (
            features["social_neighbor_count"] > 0
        ).astype("int8")

        for col in (
            "social_neighbor_mean",
            "social_neighbor_mean_min",
            "social_neighbor_mean_max",
            "social_neighbor_weighted_mean",
        ):
            features[col] = features[col].fillna(self.global_mean)

        features["social_neighbor_mean_std"] = features[
            "social_neighbor_mean_std"
        ].fillna(0)
        features["social_neighbor_mean_range"] = (
            features["social_neighbor_mean_max"] - features["social_neighbor_mean_min"]
        )
        features["social_neighbor_mean_minus_global"] = (
            features["social_neighbor_mean"] - self.global_mean
        )
        features["social_neighbor_weighted_mean_minus_global"] = (
            features["social_neighbor_weighted_mean"] - self.global_mean
        )
        features["social_user_vs_neighbor_mean"] = (
            features["user_mean"] - features["social_neighbor_weighted_mean"]
        )
        features["social_item_vs_neighbor_mean"] = (
            features["item_mean"] - features["social_neighbor_weighted_mean"]
        )

        for col in SOCIAL_FEATURES:
            features[col] = features[col].fillna(0)
        return features

    def _fill_social_item_features(self, features: pd.DataFrame) -> pd.DataFrame:
        features["social_item_neighbor_count"] = features[
            "social_item_neighbor_count"
        ].fillna(0).astype("float32")
        features["social_item_neighbor_log_count"] = np.log1p(
            features["social_item_neighbor_count"]
        )
        features["social_item_has_neighbor_rating"] = (
            features["social_item_neighbor_count"] > 0
        ).astype("int8")
        features["social_item_neighbor_share"] = (
            features["social_item_neighbor_count"]
            / features["social_degree"].replace(0, np.nan)
        )

        for col in (
            "social_item_neighbor_mean",
            "social_item_neighbor_min",
            "social_item_neighbor_max",
        ):
            features[col] = features[col].fillna(self.global_mean)

        for col in (
            "social_item_neighbor_std",
            "social_item_neighbor_frac_liked",
            "social_item_neighbor_frac_disliked",
        ):
            features[col] = features[col].fillna(0)

        features["social_item_neighbor_range"] = (
            features["social_item_neighbor_max"] - features["social_item_neighbor_min"]
        )
        features["social_item_neighbor_mean_minus_global"] = (
            features["social_item_neighbor_mean"] - self.global_mean
        )
        features["social_item_user_gap"] = (
            features["user_mean"] - features["social_item_neighbor_mean"]
        )
        features["social_item_item_gap"] = (
            features["item_mean"] - features["social_item_neighbor_mean"]
        )

        for col in SOCIAL_ITEM_FEATURES:
            features[col] = features[col].fillna(0)
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

    def evaluate(
        self,
        dataset: str,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        rebuild: bool = False,
    ) -> dict:
        social_edges = load_trust_graph(dataset)
        if social_edges is None:
            logger.info("  No social graph found for %s.", dataset)
        else:
            logger.info(
                "  Loaded social graph for %s (%s directed edges)",
                dataset,
                len(social_edges),
            )
        self._features = _FeatureEngineer(social_edges)
        self._predict_calls = 0
        return super().evaluate(dataset, train, val, test, rebuild=rebuild)

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
