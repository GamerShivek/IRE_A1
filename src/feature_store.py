"""
feature_store.py
----------------
Builds per-split article and user feature tables from the processed splits.

Output (per dataset per split):
  data/processed/{dataset}/{split}/articles_features.parquet
    - article_id, title, abstract, body, category, published_time
    - (placeholder 'embedding' column — populated by Q3 once embeddings exist)

  data/processed/{dataset}/{split}/user_features.parquet
    - user_id
    - history_article_ids : list of article_ids from click_history
    - click_count         : len(history) — recency proxy for Q4
    - (history is already bounded by the train/val split boundary from
       temporal_split.py — no future-click leakage here)
"""

import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

SPLITS = ["train", "val", "test"]
DATASETS = ["mind", "ebnerd"]


def build_article_features(split_dir: Path) -> pl.DataFrame:
    """
    articles_features = articles table (no transformation needed at this stage).
    An 'embedding' column with null values is added as a placeholder for Q3.
    """
    articles = pl.read_parquet(split_dir / "articles.parquet")

    # Add placeholder embedding column (will be filled in Q3)
    articles = articles.with_columns(
        pl.lit(None, dtype=pl.List(pl.Float32)).alias("embedding")
    )
    return articles


def build_user_features(split_dir: Path) -> pl.DataFrame:
    """
    user_features = one row per user seen in impressions or history.

    Columns:
      user_id             : str
      history_article_ids : list[str]  (from click_history, pre-boundary)
      click_count         : int        (len of history — recency proxy)
    """
    click_history = pl.read_parquet(split_dir / "click_history.parquet")
    impressions   = pl.read_parquet(split_dir / "impressions.parquet")

    # Aggregate history per user
    if len(click_history) > 0:
        user_hist = (
            click_history
            .group_by("user_id")
            .agg(
                pl.col("article_id").alias("history_article_ids"),
                pl.col("article_id").len().alias("click_count"),
            )
        )
    else:
        user_hist = pl.DataFrame(
            {"user_id": [], "history_article_ids": [], "click_count": []},
            schema={
                "user_id": pl.Utf8,
                "history_article_ids": pl.List(pl.Utf8),
                "click_count": pl.Int32,
            },
        )

    # All users who appear in impressions (cold-start users may have no history)
    all_users = impressions.select("user_id").unique()

    user_features = all_users.join(user_hist, on="user_id", how="left")
    user_features = user_features.with_columns(
        pl.col("click_count").fill_null(0).cast(pl.Int32),
        pl.col("history_article_ids").fill_null([]),
    )
    return user_features


def build_feature_store(processed_dir: Path) -> None:
    """Build article and user feature tables for every dataset/split combination."""
    summary_rows = []

    for dataset in DATASETS:
        for split in SPLITS:
            split_dir = processed_dir / dataset / split
            if not split_dir.exists():
                log.warning("Split dir not found: %s — skipping", split_dir)
                continue

            # Articles
            art_feat = build_article_features(split_dir)
            art_out  = split_dir / "articles_features.parquet"
            art_feat.write_parquet(art_out)

            # Users
            user_feat = build_user_features(split_dir)
            user_out  = split_dir / "user_features.parquet"
            user_feat.write_parquet(user_out)

            summary_rows.append({
                "dataset": dataset,
                "split": split,
                "n_articles": len(art_feat),
                "n_users": len(user_feat),
                "n_warm_users": int((user_feat["click_count"] > 0).sum()),
                "n_cold_users": int((user_feat["click_count"] == 0).sum()),
            })
            log.info(
                "[%s/%s] articles_features=%d  user_features=%d (warm=%d cold=%d)",
                dataset, split, len(art_feat), len(user_feat),
                summary_rows[-1]["n_warm_users"],
                summary_rows[-1]["n_cold_users"],
            )

    # Print summary table
    if summary_rows:
        print("\n=== Feature Store Summary ===")
        print(f"{'dataset':<10} {'split':<6} {'articles':>10} {'users':>8} {'warm':>8} {'cold':>8}")
        print("-" * 55)
        for r in summary_rows:
            print(
                f"{r['dataset']:<10} {r['split']:<6} "
                f"{r['n_articles']:>10} {r['n_users']:>8} "
                f"{r['n_warm_users']:>8} {r['n_cold_users']:>8}"
            )
    return summary_rows


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base = Path(__file__).resolve().parents[2]
    build_feature_store(processed_dir=base / "data" / "processed")
