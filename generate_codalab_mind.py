#!/usr/bin/env python3
import os
import time
import zipfile
import numpy as np
import polars as pl
from pathlib import Path
import sys
import argparse

# Allow src/ imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from retrieval.embeddings_load import load_or_compute_embeddings
from retrieval.user_vector import build_user_vector

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "predictions"
OUT_FILE = OUT_DIR / "prediction.txt"
ZIP_FILE = OUT_DIR / "mind_codalab_submission.zip"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behaviors", type=str, default=str(PROJECT_ROOT / "data/raw/mind/dev/behaviors.tsv"),
                        help="Path to the test behaviors.tsv provided by Codabench")
    args = parser.parse_args()
    
    behaviors_path = Path(args.behaviors)
    if not behaviors_path.exists():
        print(f"Error: {behaviors_path} does not exist.")
        return
    print("Loading MIND articles to compute/load embeddings...")
    # Load all unique articles across the entire dataset to ensure we have embeddings for dev
    articles_df = pl.read_parquet(PROJECT_ROOT / "data/interim/mind/articles.parquet")
    
    # load_or_compute_embeddings expects article_id to be 'mind_N123'
    # We will get back a list of IDs and a matrix
    article_ids, vectors = load_or_compute_embeddings("mind", articles_df)
    
    # Build dictionary for fast lookup
    print("Building embedding dictionary...")
    emb_dict = {aid: vec for aid, vec in zip(article_ids, vectors)}
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Processing behaviors.tsv line-by-line...")
    with open(behaviors_path, "r", encoding="utf-8") as fin, open(OUT_FILE, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.strip("\n").split("\t")
            if len(parts) < 5:
                continue
                
            imp_id = parts[0]
            # history is space separated, but we need to prefix with 'mind_'
            hist_raw = parts[3]
            history = [f"mind_{x}" for x in hist_raw.split()] if hist_raw else []
            
            # candidates are space separated like 'N123-0' 'N456-1'
            cand_raw = parts[4].split()
            candidates = [f"mind_{c.split('-')[0]}" for c in cand_raw]
            
            # 1. Build user vector
            # Get valid history embeddings
            hist_embs = [emb_dict[item] for item in history if item in emb_dict]
            
            # We use max_history = 50 as in Q3
            max_history = 50
            if hist_embs:
                # Recency weighted user vector
                hist_embs = hist_embs[-max_history:]
                n = len(hist_embs)
                weights = np.linspace(0.1, 1.0, n)
                weights /= weights.sum()
                weights = weights.reshape(-1, 1)
                
                hist_matrix = np.vstack(hist_embs)
                user_vec = np.sum(hist_matrix * weights, axis=0)
                norm = np.linalg.norm(user_vec)
                if norm > 0:
                    user_vec /= norm
            else:
                # Cold user -> zero vector
                dim = vectors.shape[1]
                user_vec = np.zeros(dim)
                
            # 2. Score candidates
            scores = []
            for c in candidates:
                if c in emb_dict:
                    score = np.dot(user_vec, emb_dict[c])
                else:
                    score = -999.0 # fallback for missing article
                scores.append(score)
                
            # 3. Rank candidates
            # We need the rank of each candidate in the original order.
            # Example: scores = [0.9, 0.1, 0.8]
            # Sorted indices (descending) = [0, 2, 1]
            # Ranks for candidates: cand 0 -> rank 1, cand 1 -> rank 3, cand 2 -> rank 2.
            # So ranks = [1, 3, 2]
            
            # Argsort descending
            sorted_indices = np.argsort(scores)[::-1]
            
            # Build rank map: index -> rank
            ranks = [0] * len(candidates)
            for rank, idx in enumerate(sorted_indices):
                ranks[idx] = rank + 1
                
            # 4. Format and write
            ranks_str = "[" + ",".join(map(str, ranks)) + "]"
            fout.write(f"{imp_id} {ranks_str}\n")

    print(f"Predictions written to {OUT_FILE}")
    
    print("Zipping directly without enclosing folders...")
    # Zip it up directly
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        # ARC name is just the file name, so it sits at root
        zf.write(OUT_FILE, arcname="prediction.txt")
        
    # Remove the txt file to keep clean
    os.remove(OUT_FILE)
    print(f"Final submission file ready at: {ZIP_FILE}")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Done in {time.time()-t0:.1f}s")
