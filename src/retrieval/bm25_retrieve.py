"""
bm25_retrieve.py
----------------
Retrieve top-K candidates per (user, impression) using a bm25s index.

Optimizations:
- Build query tokens for every unique user first.
- One vectorised bm25s.retrieve() call for ALL warm users at once.
- Fan out results from user → impressions (no re-scoring).
"""

import logging

import polars as pl
import bm25s

from retrieval.bm25_index  import _queries_to_tokenized
from retrieval.bm25_query  import build_query_tokens, build_article_title_lookup

log = logging.getLogger(__name__)

TOP_KS = [50, 100, 200]


def retrieve_for_split(
    retriever: bm25s.BM25,
    article_ids: list[str],
    vocab: dict[str, int],
    impressions: pl.DataFrame,
    user_features: pl.DataFrame,
    articles: pl.DataFrame,
    dataset: str,
    top_ks: list[int] = TOP_KS,
) -> tuple[pl.DataFrame, dict]:
    """
    Run BM25 retrieval for every impression in a split.

    All warm users are scored in a single vectorised bm25s.retrieve() call.

    Returns:
        results_df : DataFrame — impression_id, user_id, k,
                     retrieved_article_ids, clicked_article_ids, is_cold
        stats      : dict with warm/cold/no-GT counts
    """
    max_k = min(max(top_ks), len(article_ids))  # can't retrieve more than corpus size

    # ── Lookup tables ────────────────────────────────────────────────────
    title_lookup  = build_article_title_lookup(articles)
    user_hist_map = dict(zip(
        user_features["user_id"].to_list(),
        user_features["history_article_ids"].to_list(),
    ))

    # ── Build query tokens for each unique user ──────────────────────────
    unique_users = impressions["user_id"].unique().to_list()
    log.info("  Building queries for %d unique users…", len(unique_users))

    warm_users:   list[str]        = []
    warm_queries: list[list[str]]  = []
    cold_users:   set[str]         = set()

    for uid in unique_users:
        history  = user_hist_map.get(uid) or []
        q_tokens = build_query_tokens(history, title_lookup, dataset)
        if q_tokens:
            warm_users.append(uid)
            warm_queries.append(q_tokens)
        else:
            cold_users.add(uid)

    log.info("  warm users=%d  cold users=%d", len(warm_users), len(cold_users))

    # ── Batch retrieval — one call for all warm users ────────────────────
    user_top_articles: dict[str, list[str]] = {}

    if warm_queries:
        log.info("  Running batch bm25s.retrieve() for %d users…", len(warm_users))
        q_tokenized = _queries_to_tokenized(warm_queries, vocab)
        results_arr, _scores = retriever.retrieve(
            q_tokenized, k=max_k, show_progress=False
        )
        # results_arr: (n_warm_users, max_k) — corpus positions (integers)
        for uid, row in zip(warm_users, results_arr):
            user_top_articles[uid] = [article_ids[int(i)] for i in row]
        log.info("  Batch retrieval done.")

    # ── Fan out: user results → per-impression rows ──────────────────────
    rows: list[dict] = []
    n_warm_imp = n_cold_imp = n_no_gt = 0

    for imp in impressions.iter_rows(named=True):
        imp_id  = imp["impression_id"]
        uid     = imp["user_id"]
        clicked = imp["clicked_article_ids"] or []

        if not clicked:
            n_no_gt += 1
            continue

        is_cold      = uid in cold_users
        top_articles = user_top_articles.get(uid, [])

        if is_cold:
            n_cold_imp += 1
        else:
            n_warm_imp += 1

        for k in top_ks:
            rows.append({
                "impression_id":         imp_id,
                "user_id":               uid,
                "k":                     k,
                "retrieved_article_ids": top_articles[:k],
                "clicked_article_ids":   clicked,
                "is_cold":               is_cold,
            })

    results_df = pl.DataFrame(rows, schema={
        "impression_id":         pl.Utf8,
        "user_id":               pl.Utf8,
        "k":                     pl.Int32,
        "retrieved_article_ids": pl.List(pl.Utf8),
        "clicked_article_ids":   pl.List(pl.Utf8),
        "is_cold":               pl.Boolean,
    }) if rows else pl.DataFrame(schema={
        "impression_id":         pl.Utf8,
        "user_id":               pl.Utf8,
        "k":                     pl.Int32,
        "retrieved_article_ids": pl.List(pl.Utf8),
        "clicked_article_ids":   pl.List(pl.Utf8),
        "is_cold":               pl.Boolean,
    })

    stats = {
        "n_warm_impressions": n_warm_imp,
        "n_cold_impressions": n_cold_imp,
        "n_no_ground_truth":  n_no_gt,
        "n_warm_users":       len(warm_users),
        "n_cold_users":       len(cold_users),
        "total_impressions":  len(impressions),
    }
    log.info(
        "  Impressions → warm=%d  cold=%d  no_gt=%d",
        n_warm_imp, n_cold_imp, n_no_gt,
    )
    return results_df, stats
