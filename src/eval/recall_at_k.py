"""
recall_at_k.py
--------------
Dataset-agnostic recall@K evaluation.

Reused by Q2 (BM25), Q3 (embeddings), and Q4 (full harness).

recall@K for a single impression:
    |retrieved_top_K ∩ clicked_article_ids| / |clicked_article_ids|

Macro-averaged over all warm impressions (impressions with ground-truth clicks
AND a non-cold user query). Impressions with empty clicked sets are excluded
(recall is undefined there).
"""

import json
import logging
from pathlib import Path
from typing import Iterable

import polars as pl

log = logging.getLogger(__name__)


def recall_at_k_single(
    retrieved: list[str],
    ground_truth: list[str],
) -> float:
    """Recall@K for one impression. Returns 0.0 if ground_truth is empty."""
    if not ground_truth:
        return 0.0
    hits = len(set(retrieved) & set(ground_truth))
    return hits / len(ground_truth)


def compute_recall_at_k(
    results_df: pl.DataFrame,
    ks: list[int] | None = None,
    catalog_size: int | None = None,
) -> dict[int, dict]:
    """
    Compute mean recall@K from a results DataFrame.

    Args:
        results_df   : output of bm25_retrieve.retrieve_for_split — must have
                       k, retrieved_article_ids, clicked_article_ids, is_cold
        ks           : list of K values to evaluate (default: all K in results_df)
        catalog_size : total article count; if provided, adds random_baseline = K/N
                       to every per-K dict for easy comparison

    Returns:
        dict mapping K → {
            "recall":         float,  # mean recall over warm impressions
            "n_impressions":  int,
            "n_cold":         int,
            "random_baseline": float  # K/catalog_size (if catalog_size provided)
        }
    """
    if ks is None:
        ks = sorted(results_df["k"].unique().to_list())

    metrics = {}
    for k in ks:
        subset = results_df.filter((pl.col("k") == k) & (~pl.col("is_cold")))
        n_cold = int(results_df.filter(
            (pl.col("k") == k) & pl.col("is_cold")
        )["is_cold"].len())

        entry: dict = {"recall": 0.0, "n_impressions": 0, "n_cold": n_cold}

        if len(subset) > 0:
            recalls = [
                recall_at_k_single(row["retrieved_article_ids"], row["clicked_article_ids"])
                for row in subset.iter_rows(named=True)
            ]
            entry["recall"]        = round(sum(recalls) / len(recalls), 6)
            entry["n_impressions"] = len(recalls)

        if catalog_size:
            entry["random_baseline"] = round(k / catalog_size, 6)
            entry["multiple_over_random"] = (
                round(entry["recall"] / entry["random_baseline"], 2)
                if entry["random_baseline"] > 0 and entry["recall"] > 0 else 0.0
            )

        metrics[k] = entry

    return metrics



def save_recall_results(
    metrics: dict[int, dict],
    out_path: Path,
    extra: dict | None = None,
) -> None:
    """Write recall metrics to a JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"recall_at_k": metrics}
    if extra:
        payload.update(extra)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Recall@K saved to %s", out_path)


def print_recall_table(
    dataset: str,
    split: str,
    metrics: dict[int, dict],
) -> None:
    """Pretty-print recall@K table to stdout."""
    print(f"\n{'─'*50}")
    print(f" {dataset.upper()} / {split}  —  BM25 recall@K")
    print(f"{'─'*50}")
    print(f"  {'K':>6}  {'recall@K':>10}  {'impressions':>12}  {'cold':>8}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*12}  {'─'*8}")
    for k, m in sorted(metrics.items()):
        print(
            f"  {k:>6}  {m['recall']:>10.4f}  "
            f"{m['n_impressions']:>12}  {m['n_cold']:>8}"
        )
    print(f"{'─'*50}")
