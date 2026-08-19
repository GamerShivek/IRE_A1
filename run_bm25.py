#!/usr/bin/env python3
"""
run_bm25.py
-----------
Orchestrate the full BM25 candidate generation pipeline for Q2.

Steps:
  1. Sanity-check tokenizer on 5 real samples per dataset
  2. Build (or load) BM25 index per dataset
  3. For each dataset × split (val, test):
       a. Load impressions + user_features + articles from feature store
       b. Retrieve top-K candidates per impression
       c. Compute recall@K
       d. Save results to results/bm25/{dataset}/{split}/
  4. Print summary table

Usage:
    python run_bm25.py                    # both datasets, val+test
    python run_bm25.py --mind-only
    python run_bm25.py --ebnerd-only
    python run_bm25.py --force-rebuild    # ignore cached index, rebuild
    python run_bm25.py --split val        # val only
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Allow src/ imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from retrieval.tokenize    import tokenize
from retrieval.bm25_index  import load_or_build_index
from retrieval.bm25_retrieve import retrieve_for_split
from eval.recall_at_k      import compute_recall_at_k, save_recall_results, print_recall_table

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR   = PROJECT_ROOT / "results" / "bm25"
TUNING_PATH   = RESULTS_DIR / "tuning.json"

DATASET_CONFIG = {
    "mind":   {"lang": "en"},
    "ebnerd": {"lang": "da"},
}
TOP_KS = [50, 100, 200]

# Default BM25 hyperparameters (library defaults)
DEFAULT_PARAMS = {"k1": 1.5, "b": 0.75, "title_weight": 2}


def load_tuned_params(dataset: str) -> dict:
    """Load best params from tuning.json if it exists, else return defaults."""
    if TUNING_PATH.exists():
        with open(TUNING_PATH) as f:
            tuning = json.load(f)
        if dataset in tuning:
            cfg = tuning[dataset]["best_config"]
            log.info(
                "[%s] Using tuned params: k1=%.2f  b=%.2f  title_weight=%d",
                dataset, cfg["k1"], cfg["b"], cfg["title_weight"],
            )
            return cfg
    log.info("[%s] No tuning.json found — using defaults: %s", dataset, DEFAULT_PARAMS)
    return DEFAULT_PARAMS.copy()


# ── Tokenizer sanity check ───────────────────────────────────────────────────
def sanity_check_tokenizer(dataset: str) -> None:
    lang = DATASET_CONFIG[dataset]["lang"]
    articles = pl.read_parquet(
        PROCESSED_DIR / dataset / "train" / "articles_features.parquet"
    )
    print(f"\n[{dataset}] Tokenizer sanity check (lang={lang}) — 5 samples:")
    for row in articles.head(5).iter_rows(named=True):
        text = f"{row['title'] or ''} {row['abstract'] or ''}".strip()
        tokens = tokenize(text, lang=lang)
        print(f"  {text[:55]!r}")
        print(f"    → {tokens[:10]}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def run(
    datasets: list[str],
    splits: list[str],
    force_rebuild: bool = False,
    use_tuned: bool = False,
) -> dict:
    all_results = {}

    for dataset in datasets:
        log.info("=" * 55)
        log.info("Dataset: %s", dataset)
        log.info("=" * 55)

        # 1. Sanity check tokenizer
        sanity_check_tokenizer(dataset)

        # 2. Build / load index
        params = load_tuned_params(dataset) if use_tuned else DEFAULT_PARAMS
        log.info(
            "Building/loading BM25 index (k1=%.2f  b=%.2f  tw=%d)…",
            params["k1"], params["b"], params["title_weight"],
        )
        t0 = time.time()
        retriever, article_ids, id_to_idx, vocab = load_or_build_index(
            PROCESSED_DIR, dataset, force_rebuild=force_rebuild,
            k1=params["k1"], b=params["b"], title_weight=params["title_weight"],
        )
        log.info("  Index ready in %.1fs  (%d docs)", time.time() - t0, len(article_ids))

        all_results[dataset] = {}

        for split in splits:
            split_dir = PROCESSED_DIR / dataset / split
            if not split_dir.exists():
                log.warning("[%s/%s] Split dir not found — skipping", dataset, split)
                continue

            log.info("[%s/%s] Loading feature store…", dataset, split)
            impressions   = pl.read_parquet(split_dir / "impressions.parquet")
            user_features = pl.read_parquet(split_dir / "user_features.parquet")
            articles      = pl.read_parquet(split_dir / "articles_features.parquet")

            log.info(
                "  impressions=%d  users=%d  articles=%d",
                len(impressions), len(user_features), len(articles),
            )

            # 3. Retrieve
            log.info("[%s/%s] Running BM25 retrieval…", dataset, split)
            t0 = time.time()
            results_df, stats = retrieve_for_split(
                retriever=retriever,
                article_ids=article_ids,
                vocab=vocab,
                impressions=impressions,
                user_features=user_features,
                articles=articles,
                dataset=dataset,
                top_ks=TOP_KS,
            )
            log.info("  Retrieval done in %.1fs", time.time() - t0)

            # Save raw candidates (Q4 will re-consume these)
            out_dir = RESULTS_DIR / dataset / split
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
                extra={"dataset": dataset, "split": split, "retrieval": "bm25", **stats},
            )
            all_results[dataset][split] = metrics

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SUMMARY — BM25 recall@K")
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
    parser = argparse.ArgumentParser(description="BM25 lexical retrieval (Q2)")
    parser.add_argument("--mind-only",     action="store_true")
    parser.add_argument("--ebnerd-only",   action="store_true")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Rebuild BM25 index even if cached")
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument("--tuned", action="store_true",
                        help="Load best k1/b/title_weight from results/bm25/tuning.json")
    args = parser.parse_args()

    datasets = []
    if not args.ebnerd_only:
        datasets.append("mind")
    if not args.mind_only:
        datasets.append("ebnerd")

    splits = ["val", "test"] if args.split == "both" else [args.split]

    run(datasets=datasets, splits=splits,
        force_rebuild=args.force_rebuild, use_tuned=args.tuned)


if __name__ == "__main__":
    main()
