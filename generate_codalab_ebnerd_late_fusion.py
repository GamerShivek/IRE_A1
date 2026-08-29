#!/usr/bin/env python3
import os
import time
import zipfile
import numpy as np
import polars as pl
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIR = PROJECT_ROOT / "data/raw/ebnerd_testset/ebnerd_testset/test"
EMB_FILE = PROJECT_ROOT / "data/raw/google_bert_base_multilingual_cased/google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet"

OUT_DIR = PROJECT_ROOT / "predictions"
OUT_FILE = OUT_DIR / "predictions_ebnerd_late_fusion.txt"
ZIP_FILE = OUT_DIR / "ebnerd_codalab_submission_late_fusion.zip"

def main():
    print("Loading pre-computed EB-NeRD mBERT embeddings...")
    emb_df = pl.read_parquet(EMB_FILE)
    
    print("Building embedding dictionary...")
    aids = emb_df["article_id"].to_list()
    vecs = emb_df["google-bert/bert-base-multilingual-cased"].to_list()
    
    # Use float16 to save memory, and make article ids strings for consistent matching
    emb_dict = {str(aid): np.array(vec, dtype=np.float16) for aid, vec in zip(aids, vecs)}
    del emb_df, aids, vecs # Free memory immediately
    
    print("Loading history mapping...")
    history_df = pl.read_parquet(TEST_DIR / "history.parquet", columns=["user_id", "article_id_fixed"])
    
    user_hist_dict = {}
    for row in history_df.iter_rows(named=True):
        uid = row["user_id"]
        hist = row["article_id_fixed"]
        if hist:
            # Keep top 50 most recent to bound compute time
            user_hist_dict[uid] = [str(h) for h in hist[-50:]]
            
    del history_df # free memory
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Processing behaviors to generate predictions using Late Fusion...")
    t0_loop = time.time()
    processed = 0
    
    behaviors_df = pl.read_parquet(TEST_DIR / "behaviors.parquet", columns=["impression_id", "user_id", "article_ids_inview"])
    
    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        for row in behaviors_df.iter_rows(named=True):
            imp_id = row["impression_id"]
            uid = row["user_id"]
            candidates = [str(c) for c in row["article_ids_inview"]]
            
            history = user_hist_dict.get(uid, [])
            hist_embs = [emb_dict[item] for item in history if item in emb_dict]
            
            if hist_embs:
                hist_matrix = np.vstack(hist_embs)
            else:
                hist_matrix = None
                
            scores = []
            for c in candidates:
                if c in emb_dict:
                    cand_vec = emb_dict[c]
                    
                    if hist_matrix is not None:
                        # Late fusion: compute similarity between candidate and all history items
                        similarities = np.dot(hist_matrix, cand_vec)
                        
                        # Take average of Top 3 highest similarities
                        if len(similarities) >= 3:
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
            for rank_idx, cand_idx in enumerate(sorted_indices):
                ranks[cand_idx] = rank_idx + 1
                
            ranks_str = "[" + ",".join(map(str, ranks)) + "]"
            fout.write(f"{imp_id} {ranks_str}\n")
            
            processed += 1
            if processed % 100000 == 0:
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
