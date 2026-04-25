import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent.parent / "data" / "evals"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "stars"
RELEVANCE_THRESHOLD = 4.0
TOP_K = 10


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _ranking_metrics(df: pd.DataFrame, y_pred: np.ndarray) -> dict:
    tmp = df[["user_code", TARGET]].copy()
    tmp["pred"] = y_pred
    tmp["relevant"] = (tmp[TARGET] >= RELEVANCE_THRESHOLD).astype(int)

    precisions = []
    recalls = []
    ndcgs = []
    hits = []

    for _, group in tmp.groupby("user_code"):
        top_k = group.nlargest(TOP_K, "pred")
        n_relevant_in_k = top_k["relevant"].sum()
        n_relevant_total = group["relevant"].sum()

        precisions.append(n_relevant_in_k / TOP_K)

        if n_relevant_total > 0:
            recalls.append(n_relevant_in_k / n_relevant_total)
        else:
            recalls.append(0.0)

        # NDCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(top_k["relevant"]))
        ideal = sorted(group["relevant"], reverse=True)[:TOP_K]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        hits.append(1.0 if n_relevant_in_k > 0 else 0.0)

    return {
        f"precision@{TOP_K}": float(np.mean(precisions)),
        f"recall@{TOP_K}": float(np.mean(recalls)),
        f"ndcg@{TOP_K}": float(np.mean(ndcgs)),
        f"hit_rate@{TOP_K}": float(np.mean(hits)),
    }


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
    ) -> dict:
        output_path = EVAL_DIR / f"{self.name}.csv"

        if rebuild and output_path.exists():
            output_path.unlink()

        if output_path.exists():
            logger.info(f"{dataset} {self.name} already computed.")
            results = pd.read_csv(output_path).iloc[0].to_dict()
            for key, value in results.items():
                logger.info(f"  {key}: {value}")
            return results

        logger.info(f"Fitting {self.name} on {dataset}...")
        self.fit(train, val)

        val_pred = self.predict(val)
        test_pred = self.predict(test)

        val_true = val[TARGET].to_numpy()
        test_true = test[TARGET].to_numpy()

        results = {
            "dataset": dataset,
            "model": self.name,
            "val_rmse": _rmse(val_true, val_pred),
            "val_mae": _mae(val_true, val_pred),
            "test_rmse": _rmse(test_true, test_pred),
            "test_mae": _mae(test_true, test_pred),
        }
        results.update(
            {f"test_{k}": v for k, v in _ranking_metrics(test, test_pred).items()}
        )

        pd.DataFrame([results]).to_csv(output_path, index=False)

        for key, value in results.items():
            if key in ("dataset", "model"):
                continue
            logger.info(f"  {key}: {value:.4f}")

        return results
