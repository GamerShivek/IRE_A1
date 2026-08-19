"""
ebnerd_parser.py
----------------
Parses EB-NeRD demo/small parquet files into three unified-schema tables:
  - articles       : one row per article
  - impressions    : one row per impression (shown list)
  - click_history  : one row per (user, article) historical click

Uses the ebrec package if available; falls back to raw polars if not installed.

EB-NeRD behaviours.parquet has one row per *article viewed* within an
impression (not one row per impression). We group by impression_id to
reconstruct the full candidate list and clicked list.
"""

import logging
import sys
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)


def _load_articles(articles_path: Path) -> pl.DataFrame:
    """Read articles.parquet → unified articles table."""
    df = pl.read_parquet(articles_path)
    log.info("  articles.parquet: %d rows from %s", len(df), articles_path)

    articles = df.select(
        pl.col("article_id").cast(pl.Utf8).map_elements(
            lambda x: f"ebnerd_{x}", return_dtype=pl.Utf8
        ).alias("article_id"),
        pl.col("title").cast(pl.Utf8),
        pl.col("subtitle").cast(pl.Utf8).alias("abstract"),   # EB-NeRD: subtitle → abstract
        pl.col("body").cast(pl.Utf8),
        pl.col("category_str").cast(pl.Utf8).alias("category"),
        pl.col("published_time").cast(pl.Datetime),
    )
    return articles


def _load_behaviors(beh_path: Path) -> tuple[pl.DataFrame, dict]:
    """
    Read behaviors.parquet → (impressions DataFrame, stats dict).

    behaviours.parquet in EB-NeRD has one row per article shown in an
    impression session, not one row per impression. Schema includes:
      impression_id, article_id (the focal article viewed), impression_time,
      article_ids_inview (list of all candidates), article_ids_clicked (list),
      user_id, session_id, ...

    We group by impression_id to get one row per impression.
    """
    df = pl.read_parquet(beh_path)
    log.info("  behaviors.parquet: %d rows from %s", len(df), beh_path)

    # Drop rows missing user_id or impression_time
    before = len(df)
    df = df.filter(
        pl.col("user_id").is_not_null() & pl.col("impression_time").is_not_null()
    )
    dropped = before - len(df)
    if dropped:
        log.warning("  Dropped %d rows with null user_id/impression_time", dropped)
    else:
        log.info("  No rows dropped for null user_id/impression_time")

    # Prefix article IDs
    df = df.with_columns(
        pl.col("impression_id").cast(pl.Utf8).map_elements(
            lambda x: f"ebnerd_{x}", return_dtype=pl.Utf8
        ).alias("impression_id"),
        pl.col("user_id").cast(pl.Utf8).map_elements(
            lambda x: f"ebnerd_{x}", return_dtype=pl.Utf8
        ).alias("user_id"),
    )

    # article_ids_inview / article_ids_clicked are already List[Int32]
    # Prefix them too
    def _prefix_list(lst):
        if lst is None:
            return []
        return [f"ebnerd_{x}" for x in lst if x is not None]

    df = df.with_columns(
        pl.col("article_ids_inview").map_elements(
            _prefix_list, return_dtype=pl.List(pl.Utf8)
        ).alias("candidate_article_ids"),
        pl.col("article_ids_clicked").map_elements(
            _prefix_list, return_dtype=pl.List(pl.Utf8)
        ).alias("clicked_article_ids"),
    )

    # Group by impression_id → one row per impression
    # take first value of user_id, impression_time (they're the same within an impression)
    impressions = (
        df.group_by("impression_id")
        .agg(
            pl.col("user_id").first(),
            pl.col("impression_time").first(),
            pl.col("candidate_article_ids").first(),
            pl.col("clicked_article_ids").first(),
        )
        .unique(subset=["impression_id"])
    )
    log.info("  impressions (grouped): %d", len(impressions))

    stats = {"dropped_rows": dropped, "raw_rows": before, "impressions": len(impressions)}
    return impressions, stats


def _load_history(hist_path: Path) -> pl.DataFrame:
    """
    Read history.parquet → unified click_history table.

    history.parquet has one row per user with list columns:
      user_id, article_id_fixed (list), impression_time_fixed (list), ...

    We explode into one row per (user, article) pair with click_time.
    """
    df = pl.read_parquet(hist_path)
    log.info("  history.parquet: %d rows from %s", len(df), hist_path)

    df = df.with_columns(
        pl.col("user_id").cast(pl.Utf8).map_elements(
            lambda x: f"ebnerd_{x}", return_dtype=pl.Utf8
        ).alias("user_id"),
    )

    # Explode parallel lists
    df = df.select(
        pl.col("user_id"),
        pl.col("article_id_fixed").alias("article_id"),
        pl.col("impression_time_fixed").alias("click_time"),
    ).explode("article_id", "click_time")

    # Prefix article_ids
    df = df.with_columns(
        pl.col("article_id").cast(pl.Utf8).map_elements(
            lambda x: f"ebnerd_{x}" if x is not None else None,
            return_dtype=pl.Utf8,
        ).alias("article_id")
    )

    # Drop rows with null article_id
    before = len(df)
    df = df.filter(pl.col("article_id").is_not_null())
    dropped = before - len(df)
    if dropped:
        log.warning("  History: dropped %d rows with null article_id", dropped)

    df = df.unique(subset=["user_id", "article_id"])
    log.info("  click_history: %d unique (user, article) pairs", len(df))
    return df


def parse_ebnerd(
    dataset_dir: Path,
    articles_path: Path,
    out_dir: Path,
    split_name: str = "demo",
) -> dict[str, pl.DataFrame]:
    """
    Full EB-NeRD parse pipeline.

    Args:
        dataset_dir   : e.g. ebnerd_demo/ (contains train/ and validation/)
        articles_path : path to articles.parquet (at dataset_dir level)
        out_dir       : where to write interim parquet files
        split_name    : label for logging ("demo" or "small")

    Returns dict with keys: articles, impressions, click_history
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    articles = _load_articles(articles_path)

    # Collect all impressions and histories across train + validation
    all_imp, all_hist = [], []
    for sub in ["train", "validation"]:
        sub_dir = dataset_dir / sub
        if not sub_dir.exists():
            log.warning("  Subdir %s does not exist, skipping", sub_dir)
            continue
        imp, stats = _load_behaviors(sub_dir / "behaviors.parquet")
        hist = _load_history(sub_dir / "history.parquet")
        all_imp.append(imp)
        all_hist.append(hist)

    impressions   = pl.concat(all_imp).unique(subset=["impression_id"])
    click_history = pl.concat(all_hist).unique(subset=["user_id", "article_id"])

    log.info(
        "[ebnerd-%s] articles=%d  impressions=%d  click_history=%d",
        split_name, len(articles), len(impressions), len(click_history),
    )

    articles.write_parquet(out_dir / "articles.parquet")
    impressions.write_parquet(out_dir / "impressions.parquet")
    click_history.write_parquet(out_dir / "click_history.parquet")
    log.info("Interim EB-NeRD tables written to %s", out_dir)

    return {"articles": articles, "impressions": impressions, "click_history": click_history}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base = Path(__file__).resolve().parents[3]
    for name, ddir in [("demo", "ebnerd_demo"), ("small", "ebnerd_small")]:
        dataset_dir = base.parent / ddir
        articles_path = dataset_dir / "articles.parquet"
        if not articles_path.exists():
            print(f"Skipping {name}: {articles_path} not found")
            continue
        result = parse_ebnerd(
            dataset_dir=dataset_dir,
            articles_path=articles_path,
            out_dir=base / "data" / "interim" / "ebnerd",
            split_name=name,
        )
        for tname, df in result.items():
            print(f"[{name}] {tname}: {len(df)} rows")
