"""
mind_parser.py
--------------
Parses MINDsmall_train/ and MINDsmall_dev/ into three unified-schema tables:
  - articles       : one row per article
  - impressions    : one row per impression (shown list)
  - click_history  : one row per (user, article) historical click

Known dataset gaps (documented, not silently ignored):
  - MIND has NO per-article published_time → column left null.
  - MIND has NO per-click timestamps in history → click_time set to
    impression_time of the impression that contains the history list.
    This is an approximation: the history existed *before* that impression,
    but exact click times are unavailable.
  - MIND has NO article body text → body column left null.
"""

import re
import logging
import polars as pl
from pathlib import Path

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# news.tsv columns (no header row in file)
# --------------------------------------------------------------------------- #
NEWS_COLS = [
    "article_id_raw",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

# --------------------------------------------------------------------------- #
# behaviors.tsv columns (no header row in file)
# --------------------------------------------------------------------------- #
BEH_COLS = [
    "impression_id_raw",
    "user_id_raw",
    "time_str",
    "history_str",
    "impressions_str",
]


def _parse_news(path: Path) -> pl.DataFrame:
    """Read news.tsv → unified articles table."""
    df = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=NEWS_COLS,
        quote_char=None,           # MIND TSV has unescaped quotes — disable quoting
        schema_overrides={c: pl.Utf8 for c in NEWS_COLS},
        null_values=[""],
        truncate_ragged_lines=True,
    )
    log.info("  news.tsv: %d rows loaded from %s", len(df), path)

    articles = df.select(
        pl.col("article_id_raw").cast(pl.Utf8).map_elements(
            lambda x: f"mind_{x}", return_dtype=pl.Utf8
        ).alias("article_id"),
        pl.col("title").cast(pl.Utf8),
        pl.col("abstract").cast(pl.Utf8).alias("abstract"),
        pl.lit(None, dtype=pl.Utf8).alias("body"),       # MIND: no body
        pl.col("category").cast(pl.Utf8),
        pl.lit(None, dtype=pl.Datetime).alias("published_time"),  # MIND: no timestamp
    )
    return articles


def _parse_behaviors(path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Read behaviors.tsv → unified (impressions, click_history) tables.

    History limitation: MIND embeds history as a space-separated list of
    article IDs with *no individual timestamps*.  We record the impression's
    own timestamp as a proxy click_time and flag this in the column description.
    """
    df = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=BEH_COLS,
        quote_char=None,           # same reason
        schema_overrides={c: pl.Utf8 for c in BEH_COLS},
        null_values=[""],
        truncate_ragged_lines=True,
    )
    log.info("  behaviors.tsv: %d rows loaded from %s", len(df), path)

    # Parse impression_time
    df = df.with_columns(
        pl.col("impression_id_raw").cast(pl.Utf8).map_elements(
            lambda x: f"mind_{x}", return_dtype=pl.Utf8
        ).alias("impression_id"),
        pl.col("user_id_raw").cast(pl.Utf8).map_elements(
            lambda x: f"mind_{x}", return_dtype=pl.Utf8
        ).alias("user_id"),
        pl.col("time_str").str.to_datetime(
            format="%m/%d/%Y %I:%M:%S %p", strict=False
        ).alias("impression_time"),
    )

    # ------------------------------------------------------------------
    # Drop rows with missing user_id or impression_time
    # ------------------------------------------------------------------
    before = len(df)
    df = df.filter(
        pl.col("user_id").is_not_null() & pl.col("impression_time").is_not_null()
    )
    dropped = before - len(df)
    if dropped:
        log.warning("  Dropped %d rows with null user_id or impression_time", dropped)
    else:
        log.info("  No rows dropped for null user_id/impression_time")

    # ------------------------------------------------------------------
    # Parse impressions_str → candidate_article_ids + clicked_article_ids
    # e.g.  "N55689-1 N35729-0"
    # ------------------------------------------------------------------
    def _parse_imp(s: str | None) -> dict:
        candidates, clicked = [], []
        if s:
            for token in s.split():
                parts = token.rsplit("-", 1)
                if len(parts) == 2:
                    art_id = f"mind_{parts[0]}"
                    candidates.append(art_id)
                    if parts[1] == "1":
                        clicked.append(art_id)
        return {"candidates": candidates, "clicked": clicked}

    parsed_imp = df["impressions_str"].map_elements(
        _parse_imp,
        return_dtype=pl.Struct({"candidates": pl.List(pl.Utf8), "clicked": pl.List(pl.Utf8)}),
    )
    df = df.with_columns(parsed_imp.alias("_imp_parsed"))
    df = df.with_columns(
        pl.col("_imp_parsed").struct.field("candidates").alias("candidate_article_ids"),
        pl.col("_imp_parsed").struct.field("clicked").alias("clicked_article_ids"),
    )

    # ------------------------------------------------------------------
    # Build impressions table
    # ------------------------------------------------------------------
    impressions = df.select(
        pl.col("impression_id"),
        pl.col("user_id"),
        pl.col("impression_time"),
        pl.col("candidate_article_ids"),
        pl.col("clicked_article_ids"),
    ).unique(subset=["impression_id"])

    # ------------------------------------------------------------------
    # Parse History → click_history rows
    # click_time = impression_time (approximation; see module docstring)
    # ------------------------------------------------------------------
    def _expand_history(row: dict) -> list[dict]:
        uid = row["user_id"]
        ts = row["impression_time"]
        hist = row["history_str"]
        if not hist:
            return []
        return [
            {"user_id": uid, "article_id": f"mind_{a}", "click_time": ts}
            for a in hist.split()
        ]

    history_rows = []
    for row in df.select(["user_id", "impression_time", "history_str"]).to_dicts():
        history_rows.extend(_expand_history(row))

    if history_rows:
        click_history = pl.DataFrame(history_rows).unique(subset=["user_id", "article_id"])
        log.info("  click_history: %d unique (user, article) pairs", len(click_history))
    else:
        click_history = pl.DataFrame(
            {"user_id": [], "article_id": [], "click_time": []},
            schema={"user_id": pl.Utf8, "article_id": pl.Utf8, "click_time": pl.Datetime},
        )

    return impressions, click_history


def parse_mind(
    train_dir: Path,
    dev_dir: Path,
    out_dir: Path,
) -> dict[str, pl.DataFrame]:
    """
    Full MIND parse pipeline.

    Args:
        train_dir : path to MINDsmall_train/
        dev_dir   : path to MINDsmall_dev/
        out_dir   : where to write interim parquet files (data/interim/mind/)

    Returns dict with keys: articles, impressions, click_history
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Articles (union train + dev, dedupe) -----
    arts_train = _parse_news(train_dir / "news.tsv")
    arts_dev   = _parse_news(dev_dir   / "news.tsv")
    articles   = pl.concat([arts_train, arts_dev]).unique(subset=["article_id"])
    log.info("Articles total (train+dev dedupe): %d", len(articles))

    # ----- Impressions + history (train) -----
    imp_train, hist_train = _parse_behaviors(train_dir / "behaviors.tsv")
    imp_dev,   hist_dev   = _parse_behaviors(dev_dir   / "behaviors.tsv")

    impressions   = pl.concat([imp_train, imp_dev]).unique(subset=["impression_id"])
    click_history = pl.concat([hist_train, hist_dev]).unique(subset=["user_id", "article_id"])
    log.info("Impressions total: %d", len(impressions))
    log.info("Click-history unique (user, article) pairs: %d", len(click_history))

    # ----- Write interim -----
    articles.write_parquet(out_dir / "articles.parquet")
    impressions.write_parquet(out_dir / "impressions.parquet")
    click_history.write_parquet(out_dir / "click_history.parquet")
    log.info("Interim MIND tables written to %s", out_dir)

    return {"articles": articles, "impressions": impressions, "click_history": click_history}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base = Path(__file__).resolve().parents[3]  # project root
    result = parse_mind(
        train_dir=base.parent / "MINDsmall_train",
        dev_dir=base.parent   / "MINDsmall_dev",
        out_dir=base / "data" / "interim" / "mind",
    )
    for name, df in result.items():
        print(f"{name}: {len(df)} rows, columns: {df.columns}")
