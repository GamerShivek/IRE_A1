#!/usr/bin/env python3
import argparse
import logging
import json
import numpy as np
import polars as pl
from pathlib import Path
import sys

# Allow src/ imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from eval.metrics import mrr, ndcg_at_k, auc_from_ranks, coverage, novelty, intra_list_diversity, bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

def evaluate_method(dataset: str, split: str, method: str, emb_type: str = ""):
    log.info(f"Evaluating {method} on {dataset}/{split}...")
    
    # Paths
    if method == "bm25":
        cand_path = RESULTS_DIR / "bm25" / dataset / split / "candidates.parquet"
    else:
        if dataset == "ebnerd" and emb_type:
            cand_path = RESULTS_DIR / "embeddings" / dataset / split / emb_type / "candidates.parquet"
        else:
            cand_path = RESULTS_DIR / "embeddings" / dataset / split / "candidates.parquet"
            
    if not cand_path.exists():
        log.warning(f"File {cand_path} not found. Skipping.")
        return
        
    df = pl.read_parquet(cand_path)
    
    # For evaluation we use the k=200 lists for AUC/MRR/nDCG
    df_k = df.filter(pl.col("k") == 200)
    
    # We need catalog size and popularity for novelty/coverage
    articles_df = pl.read_parquet(PROCESSED_DIR / dataset / split / "articles_features.parquet")
    catalog_size = articles_df.height
    
    # Load user features to compute article popularity (from history)
    train_users = pl.read_parquet(PROCESSED_DIR / dataset / "train" / "user_features.parquet")
    all_histories = train_users["history_article_ids"].to_list()
    pop_dict = {}
    total_interactions = 0
    for h in all_histories:
        if h:
            for item in h:
                pop_dict[item] = pop_dict.get(item, 0) + 1
                total_interactions += 1
                
    # We only evaluate on warm users for standard metrics, but for slices we can look at cold vs warm
    # Our candidates.parquet has an 'is_cold' flag.
    # Cold users were skipped (empty lists) so we can't evaluate them if they are empty.
    # We will slice by head vs tail users (based on history length) or head vs tail items?
    # Let's slice by Warm vs Cold if cold users have predictions? No, cold users have empty retrieved lists.
    
    # Load user features for slicing
    user_features = pl.read_parquet(PROCESSED_DIR / dataset / split / "user_features.parquet")
    user_hist_len = {
        row["user_id"]: len(row["history_article_ids"] or [])
        for row in user_features.iter_rows(named=True)
    }
    
    # We only evaluate on warm users for standard metrics
    valid_df = df_k.filter(~pl.col("is_cold") & (pl.col("clicked_article_ids").list.len() > 0))
    
    # Slices
    def eval_slice(df_slice, slice_name):
        if df_slice.height == 0:
            return
            
        mrr_list = []
        auc_list = []
        ndcg5_list = []
        ndcg10_list = []
        all_retrieved = []
        
        for row in df_slice.iter_rows(named=True):
            retrieved = row["retrieved_article_ids"]
            clicked = set(row["clicked_article_ids"])
            
            all_retrieved.append(retrieved)
            mrr_list.append(mrr(retrieved, clicked))
            auc_list.append(auc_from_ranks(retrieved, clicked, catalog_size))
            ndcg5_list.append(ndcg_at_k(retrieved, clicked, 5))
            ndcg10_list.append(ndcg_at_k(retrieved, clicked, 10))
            
        # Bootstrapped CI
        mrr_mean, mrr_lb, mrr_ub = bootstrap_ci(np.mean, mrr_list)
        auc_mean, auc_lb, auc_ub = bootstrap_ci(np.mean, auc_list)
        ndcg5_mean, ndcg5_lb, ndcg5_ub = bootstrap_ci(np.mean, ndcg5_list)
        ndcg10_mean, ndcg10_lb, ndcg10_ub = bootstrap_ci(np.mean, ndcg10_list)
        
        cov = coverage(all_retrieved, catalog_size)
        nov = novelty(all_retrieved, pop_dict, total_interactions)
        
        print(f"\n--- {slice_name} ({method.upper()} on {dataset}/{split}) ---")
        print(f"MRR:      {mrr_mean:.4f}  (95% CI: {mrr_lb:.4f} - {mrr_ub:.4f})")
        print(f"AUC:      {auc_mean:.4f}  (95% CI: {auc_lb:.4f} - {auc_ub:.4f})")
        print(f"nDCG@5:   {ndcg5_mean:.4f}  (95% CI: {ndcg5_lb:.4f} - {ndcg5_ub:.4f})")
        print(f"nDCG@10:  {ndcg10_mean:.4f}  (95% CI: {ndcg10_lb:.4f} - {ndcg10_ub:.4f})")
        print(f"Coverage: {cov:.4f}")
        print(f"Novelty:  {nov:.4f}")
        print("-------------------------------------------------\n")

    eval_slice(valid_df, "ALL WARM USERS")
    
    # Slicing: Few clicks (< 5) vs Many clicks (>= 5)
    few_clicks_uids = [u for u, l in user_hist_len.items() if l > 0 and l < 5]
    many_clicks_uids = [u for u, l in user_hist_len.items() if l >= 5]
    
    few_df = valid_df.filter(pl.col("user_id").is_in(few_clicks_uids))
    many_df = valid_df.filter(pl.col("user_id").is_in(many_clicks_uids))
    
    eval_slice(few_df, "SLICE: FEW CLICKS (<5)")
    eval_slice(many_df, "SLICE: MANY CLICKS (>=5)")

def generate_predictions(dataset: str, split: str, method: str, emb_type: str = ""):
    log.info(f"Generating Codabench prediction files for {method} on {dataset}/{split}...")
    
    if method == "bm25":
        cand_path = RESULTS_DIR / "bm25" / dataset / split / "candidates.parquet"
    else:
        if dataset == "ebnerd" and emb_type:
            cand_path = RESULTS_DIR / "embeddings" / dataset / split / emb_type / "candidates.parquet"
        else:
            cand_path = RESULTS_DIR / "embeddings" / dataset / split / "candidates.parquet"
            
    if not cand_path.exists():
        log.warning(f"File {cand_path} not found. Skipping predictions.")
        return
        
    df_k = pl.read_parquet(cand_path).filter(pl.col("k") == 200)
    impressions = pl.read_parquet(PROCESSED_DIR / dataset / split / "impressions.parquet")
    
    # We join impressions with our retrieved lists
    joined = impressions.join(df_k, on="impression_id", how="left")
    
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{dataset}_{split}_{method}_predictions.txt"
    
    with open(out_path, "w") as f:
        for row in joined.iter_rows(named=True):
            imp_id = row["impression_id"]
            # To match MIND official formatting if the IDs have 'mind_' prefix, we should strip it?
            # Or keep it as is. We'll keep it as is.
            
            candidates = row["candidate_article_ids"] or []
            retrieved = row["retrieved_article_ids"] or []
            
            # Create a lookup for rank of each candidate
            # If not in retrieved, give it rank 201 (i.e. tied at bottom)
            rank_map = {item: i+1 for i, item in enumerate(retrieved)}
            
            # Sort candidates by their rank in our retrieved list
            # We want smaller rank first. If not found, rank = 99999
            scored_candidates = sorted(candidates, key=lambda x: rank_map.get(x, 99999))
            
            # Format: imp_id [rank1,rank2,...] where rank1 is the rank of the 1st candidate in the original list
            # Wait, MIND format: impression_ID [rank_of_cand_1, rank_of_cand_2, ...]
            # E.g. if candidate_article_ids is [A, B, C] and our ranking is B, A, C
            # Then ranks are: A=2, B=1, C=3. String would be "1 2,1,3" -> no, just "[2,1,3]"
            
            # Let's output ranks corresponding to original candidate order
            final_ranks = []
            # We need to map our sorted list back to ranks 1..N
            cand_to_final_rank = {item: i+1 for i, item in enumerate(scored_candidates)}
            
            for c in candidates:
                final_ranks.append(str(cand_to_final_rank[c]))
                
            ranks_str = "[" + ",".join(final_ranks) + "]"
            
            # For MIND specifically, it's often imp_id \t rank1,rank2,... without brackets, but let's do brackets if generic.
            # Actually standard MIND is imp_id \t [1,3,2,...]
            imp_id_clean = imp_id.replace(f"{dataset}_", "") if dataset == "mind" else imp_id
            f.write(f"{imp_id_clean} {ranks_str}\n")
            
    log.info(f"Predictions saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true", help="Generate prediction files for codabench")
    args = parser.parse_args()
    
    methods = ["bm25", "embeddings"]
    datasets = ["mind", "ebnerd"]
    splits = ["val", "test"]
    
    for method in methods:
        for dataset in datasets:
            for split in splits:
                emb_type = "mbert" if dataset == "ebnerd" and method == "embeddings" else ""
                evaluate_method(dataset, split, method, emb_type)
                if args.predict:
                    generate_predictions(dataset, split, method, emb_type)
