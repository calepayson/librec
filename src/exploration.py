import ast
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).parent.parent / "data"

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

COLORS = {"lthing": "#4C72B0", "epinions": "#DD8452"}


def _savefig(path: Path) -> None:
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()


def _gini(values: np.ndarray) -> float:
    v = np.sort(values.astype(float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    return float((2 * (np.arange(1, n + 1) * v).sum() / (n * v.sum())) - (n + 1) / n)


# -- Data loading --------------------------------------------------------------


def _load_lthing() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    with open(DATA_DIR / "lthing_data" / "reviews.txt") as f:
        for line in tqdm(f, desc="loading lthing"):
            line = line.strip()
            if not line.startswith("reviews["):
                continue
            try:
                sep = line.index(" = ")
                records.append(ast.literal_eval(line[sep + 3 :]))
            except (ValueError, SyntaxError):
                continue

    edge_rows = []
    with open(DATA_DIR / "lthing_data" / "edges.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                edge_rows.append({"src": parts[0], "dst": parts[1]})

    return pd.DataFrame(records), pd.DataFrame(edge_rows)


def _load_epinions() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    with open(DATA_DIR / "epinions_data" / "epinions.txt", encoding="latin-1") as f:
        next(f)
        for line in tqdm(f, desc="loading epinions"):
            parts = line.strip().split(None, 5)
            if len(parts) < 5:
                continue
            try:
                records.append(
                    {
                        "item": parts[0],
                        "user": parts[1],
                        "paid": float(parts[2]),
                        "time": int(parts[3]),
                        "stars": float(parts[4]),
                        "words": parts[5] if len(parts) > 5 else "",
                    }
                )
            except ValueError:
                continue

    trust_rows = []
    with open(DATA_DIR / "epinions_data" / "network_trust.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                try:
                    trust_rows.append(
                        {"src": parts[0], "dst": parts[1], "weight": float(parts[2])}
                    )
                except ValueError:
                    trust_rows.append({"src": parts[0], "dst": parts[1], "weight": 1.0})

    df = pd.DataFrame(records)
    df = df[df["stars"].between(1, 5)].reset_index(drop=True)
    return df, pd.DataFrame(trust_rows)


# -- Basic stats ---------------------------------------------------------------


def lthing_stats(df: pd.DataFrame, edges: pd.DataFrame, rebuild: bool = False) -> None:
    """Write summary statistics for the LibraryThing dataset to a file."""
    output_path = OUTPUT_DIR / "lthing_stats.txt"
    plot_path = OUTPUT_DIR / "lthing_ratings.png"
    if rebuild:
        for p in (output_path, plot_path):
            if p.exists():
                p.unlink()
    if output_path.exists():
        logger.info("lthing stats already computed.")
        for line in output_path.read_text().splitlines():
            logger.info(line)
        return

    logger.info("Computing lthing stats... (~1.7M lines)")
    n_edges = len(edges)
    star_dist = df["stars"].dropna().value_counts().sort_index()
    sparsity = len(df) / (df["user"].nunique() * df["work"].nunique()) * 100

    lines = [
        "=== LibraryThing ===",
        f"  Reviews:      {len(df):,}",
        f"  Users:        {df['user'].nunique():,}",
        f"  Works:        {df['work'].nunique():,}",
        f"  Social edges: {n_edges:,}",
        f"  Sparsity:     {sparsity:.4f}%",
        "  Features:",
        *[f"    {col:<12}{str(dtype):<10}" for col, dtype in df.dtypes.items()],
        "  Rating distribution:",
        *[f"    {stars:>4}: {count:,}" for stars, count in star_dist.items()],
    ]
    output_path.write_text("\n".join(lines) + "\n")
    star_dist.plot.bar(
        title="LibraryThing Rating Distribution", xlabel="Stars", ylabel="Reviews"
    )
    _savefig(OUTPUT_DIR / "lthing_ratings.png")
    for line in lines:
        logger.info(line)


def epinions_stats(
    df: pd.DataFrame, trust: pd.DataFrame, rebuild: bool = False
) -> None:
    """Write summary statistics for the Epinions dataset to a file."""
    output_path = OUTPUT_DIR / "epinions_stats.txt"
    plot_path = OUTPUT_DIR / "epinions_ratings.png"
    if rebuild:
        for p in (output_path, plot_path):
            if p.exists():
                p.unlink()
    if output_path.exists():
        logger.info("Epinions stats already computed.")
        for line in output_path.read_text().splitlines():
            logger.info(line)
        return

    logger.info("Computing epinions stats... (~195K lines)")
    n_trust = len(trust)
    star_dist = df["stars"].value_counts().sort_index()
    sparsity = len(df) / (df["user"].nunique() * df["item"].nunique()) * 100

    lines = [
        "=== Epinions ===",
        f"  Reviews:     {len(df):,}",
        f"  Users:       {df['user'].nunique():,}",
        f"  Items:       {df['item'].nunique():,}",
        f"  Trust edges: {n_trust:,}",
        f"  Sparsity:    {sparsity:.4f}%",
        "  Features:",
        *[f"    {col:<12}{str(dtype):<10}" for col, dtype in df.dtypes.items()],
        "  Rating distribution:",
        *[f"    {stars:>4}: {count:,}" for stars, count in star_dist.items()],
    ]
    output_path.write_text("\n".join(lines) + "\n")
    star_dist.plot.bar(
        title="Epinions Rating Distribution", xlabel="Stars", ylabel="Reviews"
    )
    _savefig(OUTPUT_DIR / "epinions_ratings.png")
    for line in lines:
        logger.info(line)


# -- EDA plots -----------------------------------------------------------------


def _plot_cold_start(df_lt: pd.DataFrame, df_ep: pd.DataFrame) -> None:
    """% of users with <= N ratings -- motivates social-aware models over vanilla CF."""
    thresholds = [1, 2, 5, 10]

    def cold_fracs(df, col):
        counts = df[col].value_counts()
        return [100 * (counts <= t).sum() / len(counts) for t in thresholds]

    lt_fracs = cold_fracs(df_lt, "user")
    ep_fracs = cold_fracs(df_ep, "user")
    x, w = np.arange(len(thresholds)), 0.35

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        b1 = ax.bar(
            x - w / 2,
            lt_fracs,
            w,
            label="LibraryThing",
            color=COLORS["lthing"],
            alpha=0.9,
        )
        b2 = ax.bar(
            x + w / 2,
            ep_fracs,
            w,
            label="Epinions",
            color=COLORS["epinions"],
            alpha=0.9,
        )
        ax.bar_label(b1, fmt="%.1f%%", padding=3, fontsize=9)
        ax.bar_label(b2, fmt="%.1f%%", padding=3, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"<= {t} ratings" for t in thresholds])
        ax.set_ylabel("% of all users")
        ax.set_title("Cold-Start Severity by Dataset", fontweight="bold")
        ax.legend()
        ax.set_ylim(0, max(max(lt_fracs), max(ep_fracs)) + 12)
        fig.tight_layout()
        _savefig(OUTPUT_DIR / "cold_start_severity.png")


def _plot_social_coverage(
    df_lt: pd.DataFrame,
    edges_lt: pd.DataFrame,
    df_ep: pd.DataFrame,
    trust_ep: pd.DataFrame,
) -> None:
    """% of cold users with >= 1 social/trust edge -- the case for trust propagation."""
    thresholds = [1, 2, 5, 10, 20]

    def coverage(df, edges, user_col):
        if edges.empty or "src" not in edges.columns:
            return [0.0] * len(thresholds)
        social = set(edges["src"].unique())
        counts = df[user_col].value_counts()
        return [
            100
            * len(set(counts[counts <= t].index) & social)
            / max(len(counts[counts <= t]), 1)
            for t in thresholds
        ]

    lt_cov = coverage(df_lt, edges_lt, "user")
    ep_cov = coverage(df_ep, trust_ep, "user")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        for cov, marker, color, label in [
            (lt_cov, "o-", COLORS["lthing"], "LibraryThing (social friends)"),
            (ep_cov, "s-", COLORS["epinions"], "Epinions (trust edges)"),
        ]:
            ax.plot(
                thresholds,
                cov,
                marker,
                color=color,
                linewidth=2,
                markersize=7,
                label=label,
            )
            for xv, yv in zip(thresholds, cov):
                ax.annotate(
                    f"{yv:.0f}%",
                    (xv, yv),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=9,
                    color=color,
                )
        ax.set_xticks(thresholds)
        ax.set_xticklabels([f"<= {t} ratings" for t in thresholds])
        ax.set_ylabel("% of cold users with >= 1 social connection")
        ax.set_title("Social Coverage of Cold-Start Users", fontweight="bold")
        ax.legend()
        ax.set_ylim(0, 110)
        fig.tight_layout()
        _savefig(OUTPUT_DIR / "social_coverage_cold_users.png")


def _plot_trust_overlap(df_ep: pd.DataFrame, trust_ep: pd.DataFrame) -> None:
    """Epinions: co-rated items per trust pair -- low overlap justifies using the trust graph."""
    if trust_ep.empty or "src" not in trust_ep.columns:
        logger.warning("Skipping trust overlap: no trust data.")
        return

    user_items = df_ep.groupby("user")["item"].apply(set).to_dict()
    pairs = trust_ep[["src", "dst"]].drop_duplicates()
    if len(pairs) > 30_000:
        pairs = pairs.sample(30_000, random_state=42)

    counts = np.array(
        [
            len(user_items.get(s, set()) & user_items.get(d, set()))
            for s, d in tqdm(
                pairs.itertuples(index=False), total=len(pairs), desc="trust overlap"
            )
        ]
    )
    zero_pct = 100 * (counts == 0).sum() / len(counts)

    max_overlap = int(counts.max())

    with plt.rc_context(STYLE):
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4))
        fig.suptitle(
            "Epinions: Co-Rating Overlap Between Trusted Pairs",
            fontsize=13,
            fontweight="bold",
        )
        fig.subplots_adjust(top=0.88)

        max_show = min(max_overlap, 30)
        bins = range(0, max(max_show + 2, 3))
        ax.hist(
            counts[counts <= max_show],
            bins=bins,
            color=COLORS["epinions"],
            alpha=0.85,
            edgecolor="white",
            align="left",
        )
        ax.set_xlabel("# items co-rated per trust pair")
        ax.set_ylabel("Number of trust pairs")
        ax.set_title("Distribution", fontweight="bold")
        ax.text(
            0.6,
            0.8,
            f"{zero_pct:.1f}% share\n0 items",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"
            ),
        )

        sorted_counts = np.sort(counts)
        ax2.plot(
            sorted_counts,
            np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100,
            color=COLORS["epinions"],
            linewidth=2,
        )
        ax2.set_xlabel("# co-rated items per trust pair")
        ax2.set_ylabel("Cumulative % of trust pairs")
        ax2.set_title("CDF", fontweight="bold")
        xlim_max = max(max_overlap, 1)
        ax2.set_xlim(0, min(xlim_max, 50))
        for k in [0, 1, 5]:
            if k <= xlim_max:
                ax2.axvline(k, linestyle="--", color="gray", linewidth=0.9, alpha=0.7)
                ax2.text(
                    k + 0.4,
                    15 + k * 8,
                    f"{100 * (counts <= k).mean():.0f}% <= {k}",
                    fontsize=8.5,
                    color="gray",
                )

        _savefig(OUTPUT_DIR / "epinions_trust_rating_overlap.png")


def _plot_lorenz(df_lt: pd.DataFrame, df_ep: pd.DataFrame) -> None:
    """Lorenz curves + Gini -- high Gini motivates BPR/WARP over MSE."""

    def lorenz(series):
        v = np.sort(series.values.astype(float))
        return np.linspace(0, 1, len(v) + 1), np.concatenate(
            [[0], np.cumsum(v) / v.sum()]
        )

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            "Lorenz Curves: Inequality of Rating Contributions",
            fontsize=13,
            fontweight="bold",
        )

        for ax, (name, df, user_col, item_col, color) in zip(
            axes,
            [
                ("LibraryThing", df_lt, "user", "work", COLORS["lthing"]),
                ("Epinions", df_ep, "user", "item", COLORS["epinions"]),
            ],
        ):
            u, i = df[user_col].value_counts(), df[item_col].value_counts()
            px, py = lorenz(u)
            ix, iy = lorenz(i)
            ax.plot(
                px,
                py,
                color=color,
                linewidth=2,
                label=f"Users  (Gini={_gini(u.values):.3f})",
            )
            ax.plot(
                ix,
                iy,
                color=color,
                linewidth=2,
                linestyle="--",
                label=f"Items  (Gini={_gini(i.values):.3f})",
            )
            ax.plot(
                [0, 1],
                [0, 1],
                "k--",
                linewidth=0.8,
                alpha=0.4,
                label="Perfect equality",
            )
            ax.fill_between(px, px, py, alpha=0.08, color=color)
            ax.set_title(name, fontweight="bold")
            ax.set_xlabel("Cumulative fraction of users / items")
            ax.set_ylabel("Cumulative fraction of ratings")
            ax.legend(fontsize=9)

        fig.tight_layout()
        _savefig(OUTPUT_DIR / "lorenz_curves.png")


def _write_comparison_table(
    df_lt: pd.DataFrame,
    edges_lt: pd.DataFrame,
    df_ep: pd.DataFrame,
    trust_ep: pd.DataFrame,
) -> None:
    """Side-by-side dataset comparison table for the report."""

    def stats(df, user_col, item_col, edges):
        counts, i_counts = df[user_col].value_counts(), df[item_col].value_counts()
        social = (
            set(edges["src"].unique())
            if (not edges.empty and "src" in edges.columns)
            else set()
        )
        cold_set = set(counts[counts <= 5].index)
        return {
            "Reviews": f"{len(df):,}",
            "Users": f"{df[user_col].nunique():,}",
            "Items": f"{df[item_col].nunique():,}",
            "Social edges": f"{len(edges):,}" if not edges.empty else "N/A",
            "Sparsity (%)": f"{100 * len(df) / (df[user_col].nunique() * df[item_col].nunique()):.4f}",
            "Median user ratings": f"{int(counts.median())}",
            "Median item ratings": f"{int(i_counts.median())}",
            "Users with <=5 ratings (%)": f"{100 * (counts <= 5).sum() / len(counts):.1f}",
            "Users in social graph (%)": f"{100 * len(social & set(df[user_col].unique())) / df[user_col].nunique():.1f}",
            "Cold users w/ social (%)": f"{100 * len(cold_set & social) / len(cold_set):.1f}"
            if cold_set
            else "N/A",
            "User Gini": f"{_gini(counts.values):.3f}",
            "Item Gini": f"{_gini(i_counts.values):.3f}",
            "Mean rating": f"{df['stars'].mean():.2f}",
            "Rating std": f"{df['stars'].std():.2f}",
        }

    lt_s = stats(df_lt, "user", "work", edges_lt)
    ep_s = stats(df_ep, "user", "item", trust_ep)

    col_w = 30
    border = "=" * (31 + col_w * 2 + 5)
    lines = [
        border,
        f"| {'Metric':<28} | {'LibraryThing':^{col_w}} | {'Epinions':^{col_w}} |",
        "|" + "-" * 29 + "|" + "-" * (col_w + 2) + "|" + "-" * (col_w + 2) + "|",
        *[f"| {k:<28} | {lt_s[k]:^{col_w}} | {ep_s[k]:^{col_w}} |" for k in lt_s],
        border,
        "",
        # "Notes:",
        # "  - High Gini (>0.7) => popularity bias => prefer BPR/WARP over MSE.",
        # "  - High cold users w/ social => SocialMF / TrustSVD likely beneficial.",
        # "  - Very low sparsity => MF needs strong regularisation or side info.",
    ]

    def _truncate(df: pd.DataFrame, max_chars: int = 60) -> pd.DataFrame:
        out = df.head(2).copy()
        for col in out.select_dtypes(include="object").columns:
            out[col] = (
                out[col]
                .astype(str)
                .apply(lambda x: x[:max_chars] + "..." if len(x) > max_chars else x)
            )
        return out

    lt_head = _truncate(df_lt).T.to_string(header=False)
    ep_head = _truncate(df_ep).T.to_string(header=False)
    lines += [
        "",
        "--- LibraryThing head(2) ---",
        lt_head,
        "",
        "--- Epinions head(2) ---",
        ep_head,
    ]
    (OUTPUT_DIR / "dataset_comparison.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        logger.info(line)


# -- Entry points --------------------------------------------------------------


def eda(
    df_lt: pd.DataFrame,
    edges_lt: pd.DataFrame,
    df_ep: pd.DataFrame,
    trust_ep: pd.DataFrame,
    rebuild: bool = False,
) -> None:
    """Run EDA: cold start, social coverage, trust overlap, Lorenz curves, comparison table."""
    sentinel = OUTPUT_DIR / "eda_done.txt"
    if sentinel.exists() and not rebuild:
        logger.info("EDA already computed. Skipping.")
        return
    if rebuild and sentinel.exists():
        sentinel.unlink()

    logger.info("Running EDA...")
    _plot_cold_start(df_lt, df_ep)
    _plot_social_coverage(df_lt, edges_lt, df_ep, trust_ep)
    _plot_trust_overlap(df_ep, trust_ep)
    _plot_lorenz(df_lt, df_ep)
    _write_comparison_table(df_lt, edges_lt, df_ep, trust_ep)

    sentinel.write_text("done\n")
    logger.info("EDA complete.")


def explore(rebuild: bool = False, rebuild_eda: bool | None = None) -> None:
    """Run basic stats and EDA for all datasets."""
    if rebuild_eda is None:
        rebuild_eda = rebuild

    need_lt = rebuild or not (OUTPUT_DIR / "lthing_stats.txt").exists()
    need_ep = rebuild or not (OUTPUT_DIR / "epinions_stats.txt").exists()
    need_eda = rebuild_eda or not (OUTPUT_DIR / "eda_done.txt").exists()

    df_lt, edges_lt = _load_lthing() if (need_lt or need_eda) else (None, None)
    df_ep, trust_ep = _load_epinions() if (need_ep or need_eda) else (None, None)

    lthing_stats(df_lt, edges_lt, rebuild=rebuild)
    epinions_stats(df_ep, trust_ep, rebuild=rebuild)
    eda(df_lt, edges_lt, df_ep, trust_ep, rebuild=rebuild_eda)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    explore()
