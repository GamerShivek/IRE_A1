#!/usr/bin/env python3
import os
import time
import zipfile
import numpy as np
import polars as pl
from pathlib import Path
import sys

# Allow src/ imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from retrieval.embeddings_load import load_or_compute_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent
MIND_LARGE_DIR = PROJECT_ROOT / "data/raw/MINDlarge_test"
OUT_DIR = PROJECT_ROOT / "predictions"
OUT_FILE = OUT_DIR / "prediction_late_fusion.txt"
ZIP_FILE = OUT_DIR / "mind_large_codalab_submission_late_fusion.zip"

def parse_news(news_path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        news_path,
        separator='\t',
        has_header=False,
        new_columns=["article_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"],
        infer_schema_length=0,
        quote_char=None
    )
    df = df.with_columns(
        ("mind_" + pl.col("article_id")).alias("article_id")
    )
    return df

def main():
    if not MIND_LARGE_DIR.exists():
        print(f"Error: {MIND_LARGE_DIR} does not exist.")
        return
        
    print(f"Loading news.tsv from {MIND_LARGE_DIR}...")
    articles_df = parse_news(MIND_LARGE_DIR / "news.tsv")
    
    print(f"Loaded {len(articles_df)} articles. Computing embeddings (this may take a few minutes)...")
    article_ids, vectors = load_or_compute_embeddings("mind", articles_df)
    
    print("Building embedding dictionary...")
    emb_dict = {aid: vec for aid, vec in zip(article_ids, vectors)}
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    behaviors_path = MIND_LARGE_DIR / "behaviors.tsv"
    
    print(f"Processing behaviors.tsv from {behaviors_path} using Late Fusion...")
    
    processed = 0
    t0_loop = time.time()
    
    with open(behaviors_path, "r", encoding="utf-8") as fin, open(OUT_FILE, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.strip("\n").split("\t")
            if len(parts) < 5:
                continue
                
            imp_id = parts[0]
            hist_raw = parts[3]
            history = [f"mind_{x}" for x in hist_raw.split()] if hist_raw else []
            
            cand_raw = parts[4].split()
            candidates = [f"mind_{c.split('-')[0]}" for c in cand_raw]
            
            # Fetch history embeddings
            hist_embs = [emb_dict[item] for item in history if item in emb_dict]
            
            # Keep top 50 most recent to bound compute time
            if hist_embs:
                hist_embs = hist_embs[-50:]
                # Matrix of history items: shape (N_hist, D)
                hist_matrix = np.vstack(hist_embs)
            else:
                hist_matrix = None
                
            scores = []
            for c in candidates:
                if c in emb_dict:
                    cand_vec = emb_dict[c]
                    
                    if hist_matrix is not None:
                        # Compute similarity between candidate and ALL history items simultaneously
                        # cand_vec is (D,), hist_matrix is (N_hist, D)
                        # Resulting similarities shape is (N_hist,)
                        similarities = np.dot(hist_matrix, cand_vec)
                        
                        # Late Fusion Strategy: Take the average of the Top 3 similarity scores
                        # This rewards candidates that strongly match at least one or two specific interests
                        if len(similarities) >= 3:
                            # Partition to get top 3 efficiently
                            top_3 = np.partition(similarities, -3)[-3:]
                            score = np.mean(top_3)
                        else:
                            score = np.mean(similarities)
                    else:
                        # Cold user
                        score = -999.0
                else:
                    score = -999.0
                    
                scores.append(score)
                
            sorted_indices = np.argsort(scores)[::-1]
            ranks = [0] * len(candidates)
            for rank, idx in enumerate(sorted_indices):
                ranks[idx] = rank + 1
                
            ranks_str = "[" + ",".join(map(str, ranks)) + "]"
            fout.write(f"{imp_id} {ranks_str}\n")
            
            processed += 1
            if processed % 100000 == 0:
                elapsed = time.time() - t0_loop
                rate = processed / elapsed
                print(f"Processed {processed} lines... ({rate:.0f} lines/sec)")

    print(f"Predictions written to {OUT_FILE}")
    
    print("Zipping directly without enclosing folders...")
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUT_FILE, arcname="prediction.txt")
        
    os.remove(OUT_FILE)
    print(f"Final submission file ready at: {ZIP_FILE}")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total time: {time.time()-t0:.1f}s")
