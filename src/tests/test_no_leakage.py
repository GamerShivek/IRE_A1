"""
test_no_leakage.py
------------------
Anti-leakage assertions for Q9.

Tests:
  1. Temporal monotonicity: max(train impression_time) < min(val impression_time)
                            max(val  impression_time) < min(test impression_time)

  2. History boundary: every click_time in click_history used for val/test must be
     strictly before the val split's minimum impression_time (i.e. bounded by the
     train cutoff). Since temporal_split.py writes the same history for all splits
     (capped at val_cutoff), this check is on the history files themselves.

  3. (Spot-check) For a sample of impressions, verify that no history article was
     clicked *after* that impression's time. This is only possible for EB-NeRD
     (where history has real timestamps); MIND history lacks per-click times.

Run:
  python src/tests/test_no_leakage.py
  # or as part of the pipeline: build_pipeline.py calls this automatically.
"""

import sys
import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)


def assert_no_leakage(processed_dir: Path) -> bool:
    """
    Run all leakage checks. Returns True if all pass, False otherwise.
    Prints a clear PASS / FAIL for each check.
    """
    all_pass = True

    for dataset in ["mind", "ebnerd"]:
        ds_dir = processed_dir / dataset

        splits = {}
        for split in ["train", "val", "test"]:
            imp_path = ds_dir / split / "impressions.parquet"
            if not imp_path.exists():
                log.warning("[%s/%s] impressions.parquet not found — skipping", dataset, split)
                continue
            splits[split] = pl.read_parquet(imp_path)

        if not splits:
            log.warning("[%s] No split data found — skipping all checks", dataset)
            continue

        # ------------------------------------------------------------------ #
        # Check 1: Temporal monotonicity across splits
        # ------------------------------------------------------------------ #
        print(f"\n[{dataset}] Check 1: Temporal monotonicity")
        boundaries_ok = True
        if "train" in splits and "val" in splits:
            max_train = splits["train"]["impression_time"].max()
            min_val   = splits["val"]["impression_time"].min()
            if max_train is not None and min_val is not None:
                ok = max_train < min_val
                status = "PASS" if ok else "FAIL"
                print(f"  max_train={max_train}  min_val={min_val}  → {status}")
                if not ok:
                    boundaries_ok = False
                    all_pass = False
            else:
                print("  WARNING: could not compare (None values)")

        if "val" in splits and "test" in splits:
            max_val   = splits["val"]["impression_time"].max()
            min_test  = splits["test"]["impression_time"].min()
            if max_val is not None and min_test is not None:
                ok = max_val < min_test
                status = "PASS" if ok else "FAIL"
                print(f"  max_val={max_val}  min_test={min_test}  → {status}")
                if not ok:
                    boundaries_ok = False
                    all_pass = False
            else:
                print("  WARNING: could not compare (None values)")

        if boundaries_ok and len(splits) >= 2:
            print("  ✓ Split boundaries are temporally ordered")

        # ------------------------------------------------------------------ #
        # Check 2: History boundary — max(history click_time) must be
        # strictly less than min(impression_time) for val and test.
        #
        # This works for both datasets:
        #   EB-NeRD: history.parquet timestamps are real and predate behaviors.
        #   MIND:    click_time is approximated as impression_time of the
        #            TRAINING impressions (history is capped at val_cutoff).
        #            The approximation is valid: the last training impression
        #            ends just before the first val impression, so the check
        #            still catches any filter failure.
        # ------------------------------------------------------------------ #
        print(f"\n[{dataset}] Check 2: History boundary (max history click < min impression)")
        for split_name in ["val", "test"]:
            hist_path = ds_dir / split_name / "click_history.parquet"
            imp_path  = ds_dir / split_name / "impressions.parquet"
            if not hist_path.exists() or not imp_path.exists():
                continue

            hist = pl.read_parquet(hist_path)
            imp  = pl.read_parquet(imp_path)

            if "click_time" not in hist.columns or len(hist) == 0:
                print(f"  [{split_name}] No click_time column — skip")
                continue

            min_imp_time  = imp["impression_time"].min()
            max_hist_time = hist["click_time"].max()

            if min_imp_time is not None and max_hist_time is not None:
                ok = max_hist_time < min_imp_time
                status = "PASS" if ok else "FAIL"
                print(
                    f"  [{split_name}] max_history_click={max_hist_time}"
                    f"  min_impression={min_imp_time}  → {status}"
                )
                if not ok:
                    all_pass = False
            else:
                print(f"  [{split_name}] WARNING: None values, cannot check")

        # ------------------------------------------------------------------ #
        # Check 3: Spot-check per-impression history (EB-NeRD only — has real timestamps)
        # ------------------------------------------------------------------ #
        if dataset == "ebnerd":
            print(f"\n[{dataset}] Check 3: Per-impression spot-check (sample 1000)")
            for split_name in ["val", "test"]:
                hist_path = ds_dir / split_name / "click_history.parquet"
                imp_path  = ds_dir / split_name / "impressions.parquet"
                if not hist_path.exists() or not imp_path.exists():
                    continue

                hist = pl.read_parquet(hist_path)
                imp  = pl.read_parquet(imp_path)

                if "click_time" not in hist.columns:
                    print(f"  [{split_name}] No click_time — skipping")
                    continue

                # Sample impressions
                sample = imp.sample(min(1000, len(imp)), seed=42)

                violations = 0
                for row in sample.iter_rows(named=True):
                    uid = row["user_id"]
                    imp_time = row["impression_time"]
                    user_hist = hist.filter(pl.col("user_id") == uid)
                    if len(user_hist) == 0:
                        continue
                    max_click = user_hist["click_time"].max()
                    if max_click is not None and max_click >= imp_time:
                        violations += 1

                ok = violations == 0
                status = "PASS" if ok else f"FAIL ({violations} violations)"
                print(f"  [{split_name}] {status}")
                if not ok:
                    all_pass = False

    # ------------------------------------------------------------------ #
    # Final verdict
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 50)
    verdict = "✓ ALL LEAKAGE CHECKS PASSED" if all_pass else "✗ SOME CHECKS FAILED"
    print(verdict)
    print("=" * 50)
    return all_pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base = Path(__file__).resolve().parents[3]
    ok = assert_no_leakage(processed_dir=base / "data" / "processed")
    sys.exit(0 if ok else 1)
