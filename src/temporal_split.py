"""
temporal_split.py
-----------------
Leakage-free temporal split for both MIND and EB-NeRD unified impressions.

Strategy
--------
  1. Find min/max impression_time per dataset.
  2. Cut by date:
       test  = last TEST_DAYS days
       val   = preceding VAL_DAYS days
       train = everything before that
  3. Write per-split parquet files under data/processed/{dataset}/{train,val,test}/
  4. Write a split_manifest.json with cutoff dates and row counts.

Anti-leakage rule (enforced in test_no_leakage.py):
  A user's click history used in a given split must only contain clicks with
  click_time STRICTLY BEFORE the impression_time of that impression.
  Here we filter click_history to the train boundary when building each split's
  history table, matching the boundary used for impressions.
"""

import json
import logging
from datetime import timedelta
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

# Days for val and test windows. These are the defaults — the pipeline checks
# that each resulting split is non-empty and warns if you need to adjust.
DEFAULT_TEST_DAYS = 1
DEFAULT_VAL_DAYS  = 1


def _split_impressions(
    impressions: pl.DataFrame,
    test_days: int = DEFAULT_TEST_DAYS,
    val_days:  int = DEFAULT_VAL_DAYS,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    """
    Split impressions chronologically.

    Returns (train, val, test, info_dict).
    info_dict contains cutoff datetimes and row counts.
    """
    assert "impression_time" in impressions.columns, "impressions must have impression_time"

    t_min = impressions["impression_time"].min()
    t_max = impressions["impression_time"].max()
    log.info("  impression_time range: %s → %s", t_min, t_max)

    test_cutoff = t_max - timedelta(days=test_days)
    val_cutoff  = test_cutoff - timedelta(days=val_days)

    log.info("  val_cutoff=%s  test_cutoff=%s", val_cutoff, test_cutoff)

    train = impressions.filter(pl.col("impression_time") <= val_cutoff)
    val   = impressions.filter(
        (pl.col("impression_time") > val_cutoff) & (pl.col("impression_time") <= test_cutoff)
    )
    test  = impressions.filter(pl.col("impression_time") > test_cutoff)

    for name, df in [("train", train), ("val", val), ("test", test)]:
        if len(df) == 0:
            log.warning(
                "  Split '%s' is EMPTY — consider reducing test_days/val_days!", name
            )

    info = {
        "t_min": str(t_min),
        "t_max": str(t_max),
        "val_cutoff": str(val_cutoff),
        "test_cutoff": str(test_cutoff),
        "test_days": test_days,
        "val_days": val_days,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
    }
    return train, val, test, info


def _filter_history_for_split(
    click_history: pl.DataFrame,
    cutoff,
) -> pl.DataFrame:
    """
    Filter click_history to only include clicks strictly before `cutoff`.
    Returns the full DataFrame if cutoff is None or no click_time column exists.
    """
    if cutoff is None:
        return click_history
    if "click_time" not in click_history.columns:
        return click_history
    return click_history.filter(pl.col("click_time") < pl.lit(cutoff))


def split_dataset(
    interim_dir: Path,
    out_dir: Path,
    dataset_name: str,
    test_days: int = DEFAULT_TEST_DAYS,
    val_days:  int = DEFAULT_VAL_DAYS,
) -> dict:
    """
    Load unified tables from interim_dir, perform temporal split, write results.

    Returns the manifest dict for this dataset.
    """
    impressions   = pl.read_parquet(interim_dir / "impressions.parquet")
    articles      = pl.read_parquet(interim_dir / "articles.parquet")
    click_history = pl.read_parquet(interim_dir / "click_history.parquet")

    log.info("[%s] Loaded: impressions=%d  articles=%d  history=%d",
             dataset_name, len(impressions), len(articles), len(click_history))

    train_imp, val_imp, test_imp, info = _split_impressions(
        impressions, test_days=test_days, val_days=val_days
    )

    # ------------------------------------------------------------------ #
    # History boundary: use val_cutoff for ALL splits.
    #
    # Why a single cutoff, not progressive per split?
    # ─────────────────────────────────────────────────
    # MIND: click_time is approximated as the impression_time of the
    # containing impression.  MIND's click_history therefore spans the
    # full dataset period (train + val + test impressions all contribute
    # rows).  Giving val_hist a looser cutoff (test_cutoff) would include
    # "history" derived from val-period impressions — exactly the leakage
    # Q9 forbids.  Setting test_hist = click_history (no filter) gives the
    # entire dataset as history, which is even worse.
    #
    # EB-NeRD: history.parquet ends before the first impression, so
    # any cutoff >= min(impression_time) is a no-op — all three splits
    # will still receive the full 295,851 rows regardless.  The single
    # cutoff approach is therefore also correct for EB-NeRD.
    #
    # The conservative rule: history visible at val/test time =
    # clicks that happened strictly before the training window closed.
    # Per-impression filtering (click_time < that impression's own time)
    # is enforced at query time in BM25/embedding retrieval.
    # ------------------------------------------------------------------ #
    def _parse_cutoff(s: str):
        return pl.Series([s]).str.to_datetime(
            format="%Y-%m-%d %H:%M:%S%.f", strict=False
        )[0]

    val_cutoff_dt = _parse_cutoff(info["val_cutoff"])

    # Single boundary: all splits see only pre-val-cutoff history
    train_hist = _filter_history_for_split(click_history, val_cutoff_dt)
    val_hist   = train_hist
    test_hist  = train_hist

    # Write splits
    for split_name, imp_df, hist_df in [
        ("train", train_imp, train_hist),
        ("val",   val_imp,   val_hist),
        ("test",  test_imp,  test_hist),
    ]:
        split_dir = out_dir / dataset_name / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        articles.write_parquet(split_dir / "articles.parquet")
        imp_df.write_parquet(split_dir  / "impressions.parquet")
        hist_df.write_parquet(split_dir / "click_history.parquet")

        log.info(
            "  [%s/%s] impressions=%d  history=%d",
            dataset_name, split_name, len(imp_df), len(hist_df),
        )

    manifest = {
        "dataset": dataset_name,
        **info,
        "n_articles": len(articles),
        "n_history_raw":  len(click_history),
        "n_train_history": len(train_hist),
        "n_val_history":   len(val_hist),
        "n_test_history":  len(test_hist),
    }
    return manifest


def run_temporal_split(
    processed_dir: Path,
    interim_base: Path,
    mind_test_days:   int = DEFAULT_TEST_DAYS,
    mind_val_days:    int = DEFAULT_VAL_DAYS,
    ebnerd_test_days: int = 2,
    ebnerd_val_days:  int = 2,
) -> dict:
    """Run temporal split for both MIND and EB-NeRD."""
    manifests = {}

    mind_interim = interim_base / "mind"
    if mind_interim.exists():
        log.info("=== Temporal split: MIND ===")
        manifests["mind"] = split_dataset(
            interim_dir=mind_interim,
            out_dir=processed_dir,
            dataset_name="mind",
            test_days=mind_test_days,
            val_days=mind_val_days,
        )
    else:
        log.warning("MIND interim not found at %s — skipping", mind_interim)

    ebnerd_interim = interim_base / "ebnerd"
    if ebnerd_interim.exists():
        log.info("=== Temporal split: EB-NeRD ===")
        manifests["ebnerd"] = split_dataset(
            interim_dir=ebnerd_interim,
            out_dir=processed_dir,
            dataset_name="ebnerd",
            test_days=ebnerd_test_days,
            val_days=ebnerd_val_days,
        )
    else:
        log.warning("EB-NeRD interim not found at %s — skipping", ebnerd_interim)

    # Write manifest
    manifest_path = processed_dir / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifests, f, indent=2, default=str)
    log.info("split_manifest.json written to %s", manifest_path)

    return manifests


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base = Path(__file__).resolve().parents[2]
    result = run_temporal_split(
        processed_dir=base / "data" / "processed",
        interim_base=base / "data" / "interim",
    )
    for ds, m in result.items():
        print(f"\n[{ds}]")
        for k, v in m.items():
            print(f"  {k}: {v}")
