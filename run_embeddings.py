#!/usr/bin/env python3
"""
run_embeddings.py
-----------------
Orchestrate the full embedding-based candidate generation pipeline for Q3.

Steps:
  1. Load or compute embeddings per dataset
  2. Build BruteForceIndex
  3. For each dataset × split (val, test):
       a. Load impressions + user_features + articles from feature store
       b. Retrieve top-K candidates per impression (mean-pooled capped history)
       c. Compute recall@K
       d. Save results to results/embeddings/{dataset}/{split}/
  4. Print summary table

Usage:
    python run_embeddings.py
    python run_embeddings.py --mind-only
    python run_embeddings.py --ebnerd-only
    python run_embeddings.py --split val
    python run_embeddings.py --max-history 50
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow src/ imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from retrieval.embeddings_load import load_or_compute_embeddings
from retrieval.ann_index import build_index
from retrieval.embed_retrieve import retrieve_for_split
from eval.recall_at_k import compute_recall_at_k, save_recall_results, print_recall_table

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR   = PROJECT_ROOT / "results" / "embeddings"

DATASET_CONFIG = {
    "mind":   {"lang": "en"},
    "ebnerd": {"lang": "da"},
}
TOP_KS = [50, 100, 200]

def run(
    datasets: list[str],
    splits: list[str],
    max_history: int = 50,
    ebnerd_emb_type: str = "word2vec",
) -> dict:
    all_results = {}

    for dataset in datasets:
        log.info("=" * 55)
        log.info("Dataset: %s", dataset)
        log.info("=" * 55)

        # 1 & 2. Load embeddings and build index
        # We need the full article catalog for this. It's the same across splits,
        # so we can just load the val split's articles to get the catalog.
        val_articles_path = PROCESSED_DIR / dataset / "val" / "articles_features.parquet"
        if not val_articles_path.exists():
            # If val doesn't exist, try train
            train_articles_path = PROCESSED_DIR / dataset / "train" / "articles_features.parquet"
            if not train_articles_path.exists():
                log.warning("[%s] No articles found — skipping", dataset)
                continue
            articles_df = pl.read_parquet(train_articles_path)
        else:
            articles_df = pl.read_parquet(val_articles_path)
            
        t0 = time.time()
        article_ids, vectors = load_or_compute_embeddings(dataset, articles_df, ebnerd_emb_type=ebnerd_emb_type)
        log.info("  Embeddings ready in %.1fs", time.time() - t0)
        
        index = build_index(article_ids, vectors)

        all_results[dataset] = {}

        for split in splits:
            split_dir = PROCESSED_DIR / dataset / split
            if not split_dir.exists():
                log.warning("[%s/%s] Split dir not found — skipping", dataset, split)
                continue

            log.info("[%s/%s] Loading feature store...", dataset, split)
            impressions   = pl.read_parquet(split_dir / "impressions.parquet")
            user_features = pl.read_parquet(split_dir / "user_features.parquet")
            # We don't really need articles_df for retrieval itself since index is built, 
            # but we load it for logging count.
            articles      = pl.read_parquet(split_dir / "articles_features.parquet")

            log.info(
                "  impressions=%d  users=%d  articles=%d",
                len(impressions), len(user_features), len(articles),
            )

            # 3. Retrieve
            log.info("[%s/%s] Running Embeddings retrieval...", dataset, split)
            t0 = time.time()
            results_df, stats = retrieve_for_split(
                index=index,
                impressions=impressions,
                user_features=user_features,
                dataset=dataset,
                top_ks=TOP_KS,
                max_history=max_history,
            )
            log.info("  Retrieval done in %.1fs", time.time() - t0)

            # Save raw candidates
            out_dir = RESULTS_DIR / dataset / split
            if dataset == "ebnerd":
                out_dir = out_dir / ebnerd_emb_type
            out_dir.mkdir(parents=True, exist_ok=True)
            results_df.write_parquet(out_dir / "candidates.parquet")
            log.info("  Candidates written to %s/candidates.parquet", out_dir)

            # 4. Compute recall@K
            metrics = compute_recall_at_k(results_df, ks=TOP_KS, catalog_size=len(article_ids))
            print_recall_table(dataset, split, metrics)

            # Save JSON
            recall_path = out_dir / "recall_at_k.json"
            save_recall_results(
                metrics,
                recall_path,
                extra={"dataset": dataset, "split": split, "retrieval": "embeddings", "emb_type": ebnerd_emb_type if dataset == "ebnerd" else "minilm", "max_history": max_history, **stats},
            )
            all_results[dataset][split] = metrics

    # Summary
    print("\n" + "=" * 55)
    print("SUMMARY — Embeddings recall@K")
    print("=" * 55)
    print(f"  {'dataset':<10} {'split':<6} {'K=50':>8} {'K=100':>8} {'K=200':>8}")
    print(f"  {'─'*10} {'─'*6} {'─'*8} {'─'*8} {'─'*8}")
    for ds, splits_res in all_results.items():
        for sp, mets in splits_res.items():
            r50  = mets.get(50,  {}).get("recall", "-")
            r100 = mets.get(100, {}).get("recall", "-")
            r200 = mets.get(200, {}).get("recall", "-")
            print(
                f"  {ds:<10} {sp:<6} "
                f"{r50 if isinstance(r50, str) else f'{r50:.4f}':>8} "
                f"{r100 if isinstance(r100, str) else f'{r100:.4f}':>8} "
                f"{r200 if isinstance(r200, str) else f'{r200:.4f}':>8}"
            )
    print("=" * 55)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Embeddings semantic retrieval (Q3)")
    parser.add_argument("--mind-only",     action="store_true")
    parser.add_argument("--ebnerd-only",   action="store_true")
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument("--max-history", type=int, default=50,
                        help="Max recent history articles to mean-pool (default: 50)")
    parser.add_argument("--ebnerd-emb", choices=["word2vec", "mbert"], default="word2vec",
                        help="Which embeddings to use for EB-NeRD")
    args = parser.parse_args()

    datasets = []
    if not args.ebnerd_only:
        datasets.append("mind")
    if not args.mind_only:
        datasets.append("ebnerd")

    splits = ["val", "test"] if args.split == "both" else [args.split]

    run(datasets=datasets, splits=splits, max_history=args.max_history, ebnerd_emb_type=args.ebnerd_emb)


if __name__ == "__main__":
    main()
