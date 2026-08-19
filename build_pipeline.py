#!/usr/bin/env python3
"""
build_pipeline.py
-----------------
One-command rebuild of the entire data pipeline.

Usage:
  python build_pipeline.py               # uses defaults
  python build_pipeline.py --mind-only   # only run MIND
  python build_pipeline.py --ebnerd-only # only run EB-NeRD
  python build_pipeline.py --skip-tests  # skip anti-leakage checks

Exit codes: 0 = success, 1 = failure (missing files or leakage detected).
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent / "src"))

from parsers.mind_parser   import parse_mind
from parsers.ebnerd_parser import parse_ebnerd
from temporal_split        import run_temporal_split
from feature_store         import build_feature_store
from tests.test_no_leakage import assert_no_leakage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Path configuration ──────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
INTERIM_DIR   = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# ── Default raw data locations (all under data/raw/) ─────────────────────────
# Expected layout:
#   data/raw/mind/train/   → news.tsv, behaviors.tsv  (from MINDsmall_train.zip)
#   data/raw/mind/dev/     → news.tsv, behaviors.tsv  (from MINDsmall_dev.zip)
#   data/raw/ebnerd/demo/  → articles.parquet, train/, validation/  (ebnerd_demo.zip)
#   data/raw/ebnerd/small/ → articles.parquet, train/, validation/  (ebnerd_small.zip)
MIND_TRAIN_DIR    = RAW_DIR / "mind" / "train"
MIND_DEV_DIR      = RAW_DIR / "mind" / "dev"
EBNERD_DEMO_DIR   = RAW_DIR / "ebnerd" / "demo"
EBNERD_SMALL_DIR  = RAW_DIR / "ebnerd" / "small"


def check_raw_files(run_mind: bool, run_ebnerd: bool) -> bool:
    """Verify that required raw files exist before starting."""
    missing = []
    if run_mind:
        for p in [MIND_TRAIN_DIR / "news.tsv", MIND_TRAIN_DIR / "behaviors.tsv",
                  MIND_DEV_DIR / "news.tsv",   MIND_DEV_DIR / "behaviors.tsv"]:
            if not p.exists():
                missing.append(str(p))

    if run_ebnerd:
        demo_ok = (EBNERD_DEMO_DIR / "articles.parquet").exists()
        small_ok = (EBNERD_SMALL_DIR / "articles.parquet").exists()
        if not demo_ok and not small_ok:
            missing.append(
                f"EB-NeRD dataset: expected articles.parquet in "
                f"{EBNERD_DEMO_DIR} or {EBNERD_SMALL_DIR}"
            )

    if missing:
        log.error("Missing required raw files:")
        for m in missing:
            log.error("  %s", m)
        log.error(
            "\nPlease place raw files under data/raw/ as follows:\n"
            "  data/raw/mind/train/   → unzip MINDsmall_train.zip here\n"
            "  data/raw/mind/dev/     → unzip MINDsmall_dev.zip here\n"
            "  data/raw/ebnerd/demo/  → unzip ebnerd_demo.zip here\n"
            "  data/raw/ebnerd/small/ → unzip ebnerd_small.zip here (optional)\n"
        )
        return False
    return True


def print_summary(processed_dir: Path) -> None:
    """Print row-count summary for diffability across reruns."""
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY — row counts per table per split")
    print("=" * 60)
    for dataset in ["mind", "ebnerd"]:
        for split in ["train", "val", "test"]:
            split_dir = processed_dir / dataset / split
            if not split_dir.exists():
                continue
            counts = {}
            for fname in ["impressions.parquet", "articles.parquet",
                          "click_history.parquet", "articles_features.parquet",
                          "user_features.parquet"]:
                fpath = split_dir / fname
                if fpath.exists():
                    df = __import__("polars").read_parquet(fpath)
                    counts[fname.replace(".parquet", "")] = len(df)
            print(f"\n  [{dataset}/{split}]")
            for table, n in counts.items():
                print(f"    {table:<30} {n:>10} rows")

    manifest_path = processed_dir / "split_manifest.json"
    if manifest_path.exists():
        print("\n  split_manifest.json:")
        with open(manifest_path) as f:
            manifest = json.load(f)
        for ds, m in manifest.items():
            print(f"    [{ds}] train={m.get('n_train','-')}  val={m.get('n_val','-')}  test={m.get('n_test','-')}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Build the full data pipeline.")
    parser.add_argument("--mind-only",   action="store_true", help="Only process MIND")
    parser.add_argument("--ebnerd-only", action="store_true", help="Only process EB-NeRD")
    parser.add_argument("--skip-tests",  action="store_true", help="Skip anti-leakage tests")
    parser.add_argument("--ebnerd-dataset", choices=["demo", "small", "both"], default="demo",
                        help="Which EB-NeRD bundle to use (default: demo)")
    args = parser.parse_args()

    run_mind   = not args.ebnerd_only
    run_ebnerd = not args.mind_only

    # ── Step 1: Check raw files ─────────────────────────────────────────────
    log.info("Step 1/5: Checking raw files…")
    if not check_raw_files(run_mind, run_ebnerd):
        sys.exit(1)
    log.info("  All required raw files present ✓")

    t0 = time.time()

    # ── Step 2: Parse raw → interim ─────────────────────────────────────────
    log.info("Step 2/5: Parsing raw data → interim unified tables…")

    if run_mind:
        log.info("  Parsing MIND…")
        parse_mind(
            train_dir=MIND_TRAIN_DIR,
            dev_dir=MIND_DEV_DIR,
            out_dir=INTERIM_DIR / "mind",
        )

    if run_ebnerd:
        datasets_to_run = []
        if args.ebnerd_dataset in ("demo", "both"):
            datasets_to_run.append(("demo", EBNERD_DEMO_DIR))
        if args.ebnerd_dataset in ("small", "both"):
            datasets_to_run.append(("small", EBNERD_SMALL_DIR))

        # Use the first available dataset (they share the same interim dir)
        for name, ddir in datasets_to_run:
            articles_path = ddir / "articles.parquet"
            if not articles_path.exists():
                log.warning("  EB-NeRD %s not found at %s — skipping", name, ddir)
                continue
            log.info("  Parsing EB-NeRD (%s)…", name)
            parse_ebnerd(
                dataset_dir=ddir,
                articles_path=articles_path,
                out_dir=INTERIM_DIR / "ebnerd",
                split_name=name,
            )
            break  # Only parse one bundle into the shared interim dir

    # ── Step 3: Temporal split ──────────────────────────────────────────────
    log.info("Step 3/5: Temporal split…")
    run_temporal_split(
        processed_dir=PROCESSED_DIR,
        interim_base=INTERIM_DIR,
    )

    # ── Step 4: Feature store ───────────────────────────────────────────────
    log.info("Step 4/5: Building feature store…")
    build_feature_store(processed_dir=PROCESSED_DIR)

    # ── Step 5: Anti-leakage tests ──────────────────────────────────────────
    if not args.skip_tests:
        log.info("Step 5/5: Running anti-leakage tests…")
        ok = assert_no_leakage(processed_dir=PROCESSED_DIR)
        if not ok:
            log.error("Anti-leakage check FAILED — inspect split_manifest.json")
            sys.exit(1)
    else:
        log.info("Step 5/5: Skipped (--skip-tests)")

    elapsed = time.time() - t0
    log.info("Pipeline complete in %.1fs", elapsed)

    # ── Summary ─────────────────────────────────────────────────────────────
    print_summary(PROCESSED_DIR)


if __name__ == "__main__":
    main()
