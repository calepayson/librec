import ast
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).parent.parent / "data"


def lthing_stats() -> None:
    """Write summary statistics for the LibraryThing dataset to a file."""
    logger.info("Getting lthing exploration results...")
    output_path = OUTPUT_DIR / "lthing_stats.txt"

    if output_path.exists():
        logger.info("lthing exploration already computed.")
        for line in output_path.read_text().splitlines():
            logger.info(line)
        return

    logger.info("No lthing exploration found. Computing... (~1.7M lines)")
    reviews_path = DATA_DIR / "lthing_data" / "reviews.txt"
    edges_path = DATA_DIR / "lthing_data" / "edges.txt"

    records = []
    with open(reviews_path) as f:
        for line in tqdm(f, desc="lthing reviews"):
            line = line.strip()
            if not line.startswith("reviews["):
                continue
            try:
                sep = line.index(" = ")
                records.append(ast.literal_eval(line[sep + 3 :]))
            except (ValueError, SyntaxError):
                continue

    df = pd.DataFrame(records)
    n_edges = sum(1 for line in open(edges_path) if len(line.split()) == 2)

    star_dist = df["stars"].dropna().value_counts().sort_index()

    lines = [
        "=== LibraryThing ===",
        f"  Reviews:      {len(df):,}",
        f"  Users:        {df['user'].nunique():,}",
        f"  Works:        {df['work'].nunique():,}",
        f"  Social edges: {n_edges:,}",
        "  Rating distribution:",
        *[f"    {stars:>4}: {count:,}" for stars, count in star_dist.items()],
    ]

    output_path.write_text("\n".join(lines) + "\n")
    star_dist.plot.bar(
        title="LibraryThing Rating Distribution", xlabel="Stars", ylabel="Reviews"
    )
    plt.savefig(OUTPUT_DIR / "lthing_ratings.png", bbox_inches="tight")
    plt.close()
    for line in lines:
        logger.info(line)


def epinions_stats() -> None:
    """Write summary statistics for the Epinions dataset to a file."""
    logger.info("Getting epinions exploration results...")
    output_path = OUTPUT_DIR / "epinions_stats.txt"

    if output_path.exists():
        logger.info("Epinions exploration already computed.")
        for line in output_path.read_text().splitlines():
            logger.info(line)
        return

    logger.info("No epinions exploration found. Computing... (~195K lines)")
    reviews_path = DATA_DIR / "epinions_data" / "epinions.txt"
    trust_path = DATA_DIR / "epinions_data" / "network_trust.txt"

    records = []
    with open(reviews_path, encoding="latin-1") as f:
        next(f)  # skip header
        for line in tqdm(f, desc="epinions reviews"):
            parts = line.strip().split(None, 5)
            if len(parts) < 5:
                continue
            try:
                records.append(
                    {"item": parts[0], "user": parts[1], "stars": float(parts[4])}
                )
            except ValueError:
                continue
    df = pd.DataFrame(records)
    df = df[df["stars"].between(1, 5)]
    item_col, user_col, stars_col = "item", "user", "stars"

    n_trust = sum(1 for line in open(trust_path) if len(line.split()) == 3)
    star_dist = df[stars_col].value_counts().sort_index()

    lines = [
        "=== Epinions ===",
        f"  Reviews:     {len(df):,}",
        f"  Users:       {df[user_col].nunique():,}",
        f"  Items:       {df[item_col].nunique():,}",
        f"  Trust edges: {n_trust:,}",
        "  Rating distribution:",
        *[f"    {stars:>4}: {count:,}" for stars, count in star_dist.items()],
    ]

    output_path.write_text("\n".join(lines) + "\n")
    star_dist.plot.bar(
        title="Epinions Rating Distribution", xlabel="Stars", ylabel="Reviews"
    )
    plt.savefig(OUTPUT_DIR / "epinions_ratings.png", bbox_inches="tight")
    plt.close()
    for line in lines:
        logger.info(line)


def explore() -> None:
    """Write summary statistics for all datasets to files."""
    lthing_stats()
    epinions_stats()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    explore()
