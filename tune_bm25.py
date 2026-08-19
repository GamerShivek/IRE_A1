#!/usr/bin/env python3
"""
tune_bm25.py
------------
Grid-search BM25 hyperparameters (k1, b, title_weight) on the val split
and save the best config per dataset to results/bm25/tuning.json.

Grid:
  k1           : {0.5, 1.0, 1.5, 2.0}
  b            : {0.3, 0.5, 0.75}
  title_weight : {1, 2, 3}   (1 = equal weight, 2 = title twice, 3 = title 3×)

Objective: maximise recall@100 on the val split.

Usage:
    python tune_bm25.py               # both datasets
    python tune_bm25.py --mind-only
    python tune_bm25.py --ebnerd-only
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent / "src"))

import polars as pl
from retrieval.bm25_index   import load_or_build_index
from retrieval.bm25_retrieve import retrieve_for_split
from eval.recall_at_k       import compute_recall_at_k

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR   = PROJECT_ROOT / "results" / "bm25"

K1_GRID           = [0.5, 1.0, 1.5, 2.0]
B_GRID            = [0.3, 0.5, 0.75]
TITLE_WEIGHT_GRID = [1, 2, 3]
TUNE_K            = 100   # optimise recall@100


def tune_dataset(dataset: str) -> dict:
    log.info("=" * 55)
    log.info("Tuning: %s  (grid: %d configs)", dataset,
             len(K1_GRID) * len(B_GRID) * len(TITLE_WEIGHT_GRID))
    log.info("=" * 55)

    split_dir    = PROCESSED_DIR / dataset / "val"
    impressions  = pl.read_parquet(split_dir / "impressions.parquet")
    user_features = pl.read_parquet(split_dir / "user_features.parquet")
    articles     = pl.read_parquet(split_dir / "articles_features.parquet")

    best_recall = -1.0
    best_config = {}
    results_table = []

    configs = list(product(K1_GRID, B_GRID, TITLE_WEIGHT_GRID))
    for i, (k1, b, tw) in enumerate(configs, 1):
        t0 = time.time()
        retriever, article_ids, id_to_idx, vocab = load_or_build_index(
            PROCESSED_DIR, dataset,
            force_rebuild=False,   # uses cache if exists
            k1=k1, b=b, title_weight=tw,
        )
        results_df, _ = retrieve_for_split(
            retriever=retriever,
            article_ids=article_ids,
            vocab=vocab,
            impressions=impressions,
            user_features=user_features,
            articles=articles,
            dataset=dataset,
            top_ks=[TUNE_K],
        )
        metrics  = compute_recall_at_k(results_df, ks=[TUNE_K],
                                        catalog_size=len(article_ids))
        recall   = metrics[TUNE_K]["recall"]
        elapsed  = time.time() - t0

        cfg = {"k1": k1, "b": b, "title_weight": tw}
        results_table.append({**cfg, f"recall@{TUNE_K}": recall})
        log.info(
            "  [%2d/%d] k1=%.2f  b=%.2f  tw=%d  recall@%d=%.5f  (%.1fs)",
            i, len(configs), k1, b, tw, TUNE_K, recall, elapsed,
        )

        if recall > best_recall:
            best_recall = recall
            best_config = cfg

    log.info("Best for %s: %s  recall@%d=%.5f", dataset, best_config, TUNE_K, best_recall)
    return {
        "dataset":     dataset,
        "best_config": best_config,
        f"best_recall@{TUNE_K}": best_recall,
        "grid_results": results_table,
    }


def main():
    parser = argparse.ArgumentParser(description="BM25 hyperparameter tuning")
    parser.add_argument("--mind-only",   action="store_true")
    parser.add_argument("--ebnerd-only", action="store_true")
    args = parser.parse_args()

    datasets = []
    if not args.ebnerd_only:
        datasets.append("mind")
    if not args.mind_only:
        datasets.append("ebnerd")

    all_results = {}
    best_configs = {}

    for dataset in datasets:
        result = tune_dataset(dataset)
        all_results[dataset] = result
        best_configs[dataset] = result["best_config"]

    # Save tuning results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tuning_path = RESULTS_DIR / "tuning.json"
    with open(tuning_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Tuning results saved to %s", tuning_path)

    # Print summary table
    print("\n" + "=" * 60)
    print("TUNING SUMMARY")
    print("=" * 60)
    print(f"  {'dataset':<10} {'k1':>5} {'b':>6} {'tw':>4} {'recall@100':>12}")
    print(f"  {'─'*10} {'─'*5} {'─'*6} {'─'*4} {'─'*12}")
    for ds, res in all_results.items():
        cfg = res["best_config"]
        r   = res[f"best_recall@{TUNE_K}"]
        print(f"  {ds:<10} {cfg['k1']:>5.2f} {cfg['b']:>6.2f} "
              f"{cfg['title_weight']:>4}  {r:>12.5f}")
    print("=" * 60)

    # Emit best configs for use in run_bm25.py
    print("\nBest configs (copy into run_bm25.py DATASET_CONFIG or use --tuned flag):")
    for ds, cfg in best_configs.items():
        print(f"  {ds}: k1={cfg['k1']}, b={cfg['b']}, title_weight={cfg['title_weight']}")


if __name__ == "__main__":
    main()
