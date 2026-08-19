import logging
import polars as pl
import numpy as np

from retrieval.user_vector import build_user_vector

log = logging.getLogger(__name__)

TOP_KS = [50, 100, 200]

def retrieve_for_split(
    index,
    impressions: pl.DataFrame,
    user_features: pl.DataFrame,
    dataset: str,
    top_ks: list[int] = TOP_KS,
    max_history: int = 50,
) -> tuple[pl.DataFrame, dict]:
    """
    Run embedding retrieval for every impression in a split.
    """
    max_k = min(max(top_ks), len(index.article_ids))

    # user history map
    user_hist_map = dict(zip(
        user_features["user_id"].to_list(),
        user_features["history_article_ids"].to_list(),
    ))

    unique_users = impressions["user_id"].unique().to_list()
    log.info("  Building query vectors for %d unique users...", len(unique_users))

    warm_users: list[str] = []
    warm_vectors: list[np.ndarray] = []
    cold_users: set[str] = set()

    for uid in unique_users:
        history = user_hist_map.get(uid) or []
        u_vec = build_user_vector(history, index, max_history=max_history)
        if u_vec is not None:
            warm_users.append(uid)
            warm_vectors.append(u_vec)
        else:
            cold_users.add(uid)

    log.info("  warm users=%d  cold users=%d", len(warm_users), len(cold_users))

    user_top_articles: dict[str, list[str]] = {}

    if warm_vectors:
        log.info("  Running batch index search for %d users...", len(warm_users))
        q_matrix = np.vstack(warm_vectors)
        # scores: unused, indices: top max_k
        distances, indices = index.search(q_matrix, k=max_k)
        
        for i, uid in enumerate(warm_users):
            user_top_articles[uid] = [index.article_ids[idx] for idx in indices[i]]
            
        log.info("  Batch retrieval done.")

    # Fan out
    rows: list[dict] = []
    n_warm_imp = n_cold_imp = n_no_gt = 0

    for imp in impressions.iter_rows(named=True):
        imp_id = imp["impression_id"]
        uid = imp["user_id"]
        clicked = imp["clicked_article_ids"] or []

        if not clicked:
            n_no_gt += 1
            continue

        is_cold = uid in cold_users
        top_articles = user_top_articles.get(uid, [])

        if is_cold:
            n_cold_imp += 1
        else:
            n_warm_imp += 1

        for k in top_ks:
            rows.append({
                "impression_id": imp_id,
                "user_id": uid,
                "k": k,
                "retrieved_article_ids": top_articles[:k],
                "clicked_article_ids": clicked,
                "is_cold": is_cold,
            })

    results_df = pl.DataFrame(rows, schema={
        "impression_id": pl.Utf8,
        "user_id": pl.Utf8,
        "k": pl.Int32,
        "retrieved_article_ids": pl.List(pl.Utf8),
        "clicked_article_ids": pl.List(pl.Utf8),
        "is_cold": pl.Boolean,
    }) if rows else pl.DataFrame(schema={
        "impression_id": pl.Utf8,
        "user_id": pl.Utf8,
        "k": pl.Int32,
        "retrieved_article_ids": pl.List(pl.Utf8),
        "clicked_article_ids": pl.List(pl.Utf8),
        "is_cold": pl.Boolean,
    })

    stats = {
        "n_warm_impressions": n_warm_imp,
        "n_cold_impressions": n_cold_imp,
        "n_no_ground_truth": n_no_gt,
        "n_warm_users": len(warm_users),
        "n_cold_users": len(cold_users),
        "total_impressions": len(impressions),
    }
    log.info(
        "  Impressions -> warm=%d cold=%d no_gt=%d",
        n_warm_imp, n_cold_imp, n_no_gt
    )
    return results_df, stats
