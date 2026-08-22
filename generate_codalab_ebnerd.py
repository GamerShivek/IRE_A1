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

from retrieval.ann_index import BruteForceIndex
from retrieval.user_vector import build_user_vector

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIR = PROJECT_ROOT / "data/raw/ebnerd_testset/ebnerd_testset/test"
EMB_FILE = PROJECT_ROOT / "data/raw/google_bert_base_multilingual_cased/google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet"

OUT_DIR = PROJECT_ROOT / "predictions"
OUT_FILE = OUT_DIR / "predictions_fast.txt"
ZIP_FILE = OUT_DIR / "ebnerd_codalab_submission_fast.zip"

def main():
    print("Loading pre-computed EB-NeRD mBERT embeddings...")
    emb_df = pl.read_parquet(EMB_FILE)
    
    print("Building pipeline BruteForceIndex...")
    aids = emb_df["article_id"].to_list()
    # mBERT column name
    vecs = emb_df["google-bert/bert-base-multilingual-cased"].to_list()
    
    # Use pipeline components
    index = BruteForceIndex(aids, np.array(vecs, dtype=np.float32))
    dim = index.vectors.shape[1]
    zero_vec = np.zeros(dim, dtype=np.float32)
    del emb_df, aids, vecs # Free memory immediately
    
    # 2. Build user vector dictionary directly
    print("Loading history and building user vectors (using float16 to save RAM)...")
    history_df = pl.read_parquet(TEST_DIR / "history.parquet", columns=["user_id", "article_id_fixed"])
    
    user_vec_dict = {}
    max_history = 50
    for row in history_df.iter_rows(named=True):
        uid = row["user_id"]
        hist = row["article_id_fixed"]
        # Pipeline user vector generation
        user_vec = build_user_vector(hist, index, max_history=50)
        
        if user_vec is not None:
            # Store as float16 to save 50% RAM!
            user_vec_dict[uid] = user_vec.astype(np.float16)
            
    del history_df # free memory
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Processing behaviors to generate predictions...")
    t0_loop = time.time()
    processed = 0
    
    behaviors_df = pl.read_parquet(TEST_DIR / "behaviors.parquet", columns=["impression_id", "user_id", "article_ids_inview"])
    
    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        # Stream behaviors file in chunks to prevent loading all 2.5M rows into memory
        for row in behaviors_df.iter_rows(named=True):
            imp_id = row["impression_id"]
            uid = row["user_id"]
            candidates = row["article_ids_inview"]
            
            user_vec = user_vec_dict.get(uid, zero_vec)
            
            scores = []
            for c in candidates:
                if c in index.id_to_idx:
                    idx = index.id_to_idx[c]
                    # np.dot handles float16 (user_vec) and float32 (index.vectors)
                    score = np.dot(user_vec, index.vectors[idx])
                else:
                    score = -999.0
                scores.append(score)
                
            sorted_indices = np.argsort(scores)[::-1]
            ranks = [0] * len(candidates)
            for rank_idx, cand_idx in enumerate(sorted_indices):
                ranks[cand_idx] = rank_idx + 1
                
            ranks_str = "[" + ",".join(map(str, ranks)) + "]"
            fout.write(f"{imp_id} {ranks_str}\n")
            
            processed += 1
            if processed % 500000 == 0:
                elapsed = time.time() - t0_loop
                rate = processed / elapsed
                print(f"Processed {processed} impressions... ({rate:.0f} rows/sec)")

    print(f"Predictions written to {OUT_FILE}")
    
    print("Zipping directly without enclosing folders...")
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUT_FILE, arcname="predictions.txt")
        
    os.remove(OUT_FILE)
    print(f"Final submission file ready at: {ZIP_FILE}")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total time: {time.time()-t0:.1f}s")
