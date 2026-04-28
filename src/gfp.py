import logging
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd

from base_model import BaseModel
from preprocess import load_trust_graph

logger = logging.getLogger(__name__)

# ── features ──────────────────────────────────────────────────────────────────
BASE_FEATURES = [
    "user_code", "item_code",
    "user_mean", "user_count", "user_bias",
    "item_mean", "item_count", "item_bias",
    "neighbor_mean", "social_degree",
    "days_since_start",
]
GRAPH_FEATURES = BASE_FEATURES + [
    "smoothed_item_mean",
    "smoothed_user_mean",
    "smoothed_item_mean_2hop",
    "smoothed_user_mean_2hop",
    "item_vs_smoothed",
    "user_vs_smoothed",
]
CATEGORICAL = ["user_code", "item_code"]
TARGET = "stars"

# ── lightgbm params (matched to notebook cell 15) ─────────────────────────────
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
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 20
LOG_EVERY = 50

# ── shrinkage constant ────────────────────────────────────────────────────────
K_SHRINK = 10


class GFP(BaseModel):
    """Graph Feature Propagation with Gradient Boosting.

    propagates statistical node features (mean, std, count) through the
    bipartite user-item interaction graph and the user-user social graph.
    a lightgbm regressor then learns to combine those propagated features
    to predict ratings.

    two propagation channels:
      1. bipartite (user-item): 1-hop and 2-hop mean propagation
         - smoothed_item_mean: item inherits avg rating behavior of its raters
         - smoothed_user_mean: user inherits avg quality of rated items
         - 2-hop versions extend this one step further
      2. social (user-user): 1-hop mean propagation
         - neighbor_mean: user inherits avg rating style of friends
    """

    name = "gfp"

    def __init__(self):
        self._model = None
        self._global_mean = None
        self._min_time = None

        # stored stats for predict()
        self._user_stats = None
        self._item_stats = None
        self._sm_item = None
        self._sm_user = None
        self._sm_item_2h = None
        self._sm_user_2h = None
        self._nb_mean = None
        self._nb_deg = None

        # dataset name (set via evaluate)
        self._dataset = None

    # ── override evaluate to capture dataset name ─────────────────────────────
    def evaluate(self, dataset, train, val, test, rebuild=False):
        self._dataset = dataset
        return super().evaluate(dataset, train, val, test, rebuild=rebuild)

    # ── internal helpers ──────────────────────────────────────────────────────
    def _compute_stats(self, train: pd.DataFrame) -> None:
        """compute all node statistics from train set only."""
        self._global_mean = float(train[TARGET].mean())
        self._min_time    = float(train["time"].min())

        # user stats
        u = (
            train.groupby("user_code")[TARGET]
            .agg(user_mean="mean", user_count="count", user_std="std")
            .reset_index()
            .fillna({"user_std": 0})
        )
        u["user_mean_shrunk"] = (
            u["user_count"] * u["user_mean"] + K_SHRINK * self._global_mean
        ) / (u["user_count"] + K_SHRINK)
        self._user_stats = u

        # item stats
        i = (
            train.groupby("item_code")[TARGET]
            .agg(item_mean="mean", item_count="count", item_std="std")
            .reset_index()
            .fillna({"item_std": 0})
        )
        i["item_mean_shrunk"] = (
            i["item_count"] * i["item_mean"] + K_SHRINK * self._global_mean
        ) / (i["item_count"] + K_SHRINK)
        self._item_stats = i

        logger.info(
            f"  stats: {len(u):,} users, {len(i):,} items, "
            f"global_mean={self._global_mean:.4f}"
        )

    def _compute_graph_propagation(self, train: pd.DataFrame) -> None:
        """propagate statistics through bipartite user-item graph."""
        known = train[train["item_code"] >= 0]

        u_mean = dict(zip(
            self._user_stats["user_code"],
            self._user_stats["user_mean_shrunk"]
        ))
        i_mean = dict(zip(
            self._item_stats["item_code"],
            self._item_stats["item_mean_shrunk"]
        ))

        # adjacency lists
        u2i = known.groupby("user_code")["item_code"].apply(list).to_dict()
        i2u = known.groupby("item_code")["user_code"].apply(list).to_dict()

        gm = self._global_mean

        # 1-hop
        self._sm_item = {
            i: np.mean([u_mean.get(u, gm) for u in us])
            for i, us in i2u.items()
        }
        self._sm_user = {
            u: np.mean([i_mean.get(i, gm) for i in its])
            for u, its in u2i.items()
        }

        # 2-hop
        self._sm_item_2h = {
            i: np.mean([self._sm_user.get(u, gm) for u in us])
            for i, us in i2u.items()
        }
        self._sm_user_2h = {
            u: np.mean([self._sm_item.get(i, gm) for i in its])
            for u, its in u2i.items()
        }

        logger.info(
            f"  graph propagation: {len(self._sm_item):,} items, "
            f"{len(self._sm_user):,} users covered"
        )

    def _compute_social_propagation(self, train: pd.DataFrame) -> None:
        """propagate user mean through social/trust graph."""
        trust = load_trust_graph(self._dataset)
        if trust is None or trust.empty:
            logger.info("  no trust graph found, skipping social propagation")
            self._nb_mean = {}
            self._nb_deg  = {}
            return

        u_mean = dict(zip(
            self._user_stats["user_code"],
            self._user_stats["user_mean_shrunk"]
        ))
        train_user_codes = set(self._user_stats["user_code"].tolist())

        nb_map = defaultdict(list)
        for _, row in trust.iterrows():
            nb_map[row["src"]].append(row["dst"])
            nb_map[row["dst"]].append(row["src"])

        self._nb_mean = {}
        self._nb_deg  = {}
        for uc, nbs in nb_map.items():
            vals = [u_mean[nb] for nb in nbs if nb in u_mean]
            self._nb_mean[uc] = np.mean(vals) if vals else self._global_mean
            self._nb_deg[uc]  = len(nbs)

        logger.info(
            f"  social propagation: {len(self._nb_mean):,} users with neighbors"
        )

    def _make_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """apply precomputed stats to any dataframe."""
        gm = self._global_mean
        out = df.copy()

        out["days_since_start"] = (out["time"] - self._min_time) / 86400

        out = out.merge(
            self._user_stats[["user_code", "user_mean_shrunk", "user_count"]],
            on="user_code", how="left"
        )
        out = out.merge(
            self._item_stats[["item_code", "item_mean_shrunk", "item_count"]],
            on="item_code", how="left"
        )

        out["user_mean"]  = out["user_mean_shrunk"].fillna(gm)
        out["item_mean"]  = out["item_mean_shrunk"].fillna(gm)
        out["user_count"] = out["user_count"].fillna(0)
        out["item_count"] = out["item_count"].fillna(0)
        out["user_bias"]  = out["user_mean"] - gm
        out["item_bias"]  = out["item_mean"] - gm

        uc = out["user_code"].values
        ic = out["item_code"].values

        # graph propagated features
        out["smoothed_item_mean"]     = [self._sm_item.get(i, gm) for i in ic]
        out["smoothed_user_mean"]     = [self._sm_user.get(u, gm) for u in uc]
        out["smoothed_item_mean_2hop"]= [self._sm_item_2h.get(i, gm) for i in ic]
        out["smoothed_user_mean_2hop"]= [self._sm_user_2h.get(u, gm) for u in uc]
        out["item_vs_smoothed"]       = out["item_mean"] - out["smoothed_item_mean"]
        out["user_vs_smoothed"]       = out["user_mean"] - out["smoothed_user_mean"]

        # social propagated features
        out["neighbor_mean"]  = [self._nb_mean.get(u, gm) for u in uc]
        out["social_degree"]  = [self._nb_deg.get(u, 0)   for u in uc]

        return out

    # ── BaseModel interface ───────────────────────────────────────────────────
    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        logger.info("computing node statistics...")
        self._compute_stats(train)

        logger.info("propagating features through bipartite graph...")
        self._compute_graph_propagation(train)

        logger.info("propagating features through social graph...")
        self._compute_social_propagation(train)

        logger.info("building feature matrices...")
        train_f = self._make_features(train)
        val_f   = self._make_features(val)

        train_set = lgb.Dataset(
            train_f[GRAPH_FEATURES], label=train_f[TARGET],
            categorical_feature=CATEGORICAL
        )
        val_set = lgb.Dataset(
            val_f[GRAPH_FEATURES], label=val_f[TARGET],
            categorical_feature=CATEGORICAL,
            reference=train_set
        )

        logger.info("training lightgbm...")
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
        logger.info(f"  best iteration: {self._model.best_iteration}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        feat_df = self._make_features(df)
        return self._model.predict(
            feat_df[GRAPH_FEATURES],
            num_iteration=self._model.best_iteration
        )