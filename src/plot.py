import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent.parent / "data" / "evals"
PLOT_DIR = Path(__file__).parent.parent / "data" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

RATING_METRICS = ["val_rmse", "val_mae", "test_rmse", "test_mae"]
RANKING_METRICS = [
    "test_precision@10",
    "test_recall@10",
    "test_ndcg@10",
    "test_hit_rate@10",
]

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
}


def _savefig(path: Path) -> None:
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()


def _load_evals(dataset: str) -> pd.DataFrame:
    csvs = sorted(EVAL_DIR.glob(f"{dataset}_*.csv"))
    if not csvs:
        logger.warning("No eval CSVs found for %s in %s", dataset, EVAL_DIR)
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)


def _auto_ylim(values, pad_frac=0.15):
    lo, hi = values.min(), values.max()
    span = hi - lo if hi > lo else hi * 0.1
    return lo - span * pad_frac, hi + span * pad_frac


def _plot_single_metric(df: pd.DataFrame, metric: str, dataset: str) -> None:
    path = PLOT_DIR / f"{dataset}_{metric}.png"
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(df["model"], df[metric], color="#4C72B0", alpha=0.9)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_ylim(_auto_ylim(df[metric]))
        ax.set_ylabel(metric)
        ax.set_title(f"{dataset} — {metric}", fontweight="bold")
        fig.tight_layout()
        _savefig(path)


def _plot_grouped(
    df: pd.DataFrame, metrics: list[str], dataset: str, name: str
) -> None:
    path = PLOT_DIR / f"{dataset}_{name}.png"
    present = [m for m in metrics if m in df.columns]
    if not present:
        return

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 5))
        if len(present) == 1:
            axes = [axes]
        for ax, metric in zip(axes, present):
            bars = ax.bar(df["model"], df[metric], color="#4C72B0", alpha=0.9)
            ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
            ax.set_ylim(_auto_ylim(df[metric]))
            ax.set_ylabel(metric)
            ax.set_title(metric, fontweight="bold")
            ax.tick_params(axis="x", rotation=45)
        fig.suptitle(
            f"{dataset} — {name.replace('_', ' ').title()}",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        _savefig(path)


def plot(dataset: str, rebuild: bool = False) -> None:
    sentinel = PLOT_DIR / f"{dataset}_plots_done.txt"

    if rebuild and sentinel.exists():
        for p in PLOT_DIR.glob(f"{dataset}_*.png"):
            p.unlink()
        sentinel.unlink()

    if sentinel.exists():
        logger.info("Plots for %s already generated.", dataset)
        return

    df = _load_evals(dataset)
    if df.empty:
        return

    for metric in RATING_METRICS + RANKING_METRICS:
        if metric in df.columns:
            _plot_single_metric(df, metric, dataset)
            logger.info(f"  Saved {dataset}_{metric}.png")

    _plot_grouped(df, RATING_METRICS, dataset, "rating_metrics")
    logger.info(f"  Saved {dataset}_rating_metrics.png")

    _plot_grouped(df, RANKING_METRICS, dataset, "ranking_metrics")
    logger.info(f"  Saved {dataset}_ranking_metrics.png")

    sentinel.write_text("done\n")
    logger.info("All plots for %s saved to %s", dataset, PLOT_DIR)
