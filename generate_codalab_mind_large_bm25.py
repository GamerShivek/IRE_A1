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

from retrieval.bm25_index import build_index
from retrieval.tokenize import tokenize_batch

PROJECT_ROOT = Path(__file__).resolve().parent
MIND_LARGE_DIR = PROJECT_ROOT / "data/raw/MINDlarge_test"
OUT_DIR = PROJECT_ROOT / "predictions"
OUT_FILE = OUT_DIR / "prediction_bm25.txt"
ZIP_FILE = OUT_DIR / "mind_large_codalab_submission_bm25.zip"

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
    
    print(f"Loaded {len(articles_df)} articles. Building BM25 index (this is fast)...")
    # build_index automatically caches to disk based on hyperparameters, so it won't rebuild if already done
    retriever, article_ids, id_to_idx, vocab = build_index(articles_df, "mind", force_rebuild=False)
    
    print("Building article title token dictionary for fast query generation...")
    # BM25 queries in our pipeline use the article titles of the history
    titles = articles_df["title"].fill_null("").to_list()
    aids = articles_df["article_id"].to_list()
    
    # Tokenize all titles
    tokenized_titles = tokenize_batch(titles, lang="en")
    
    # Map token strings to vocab IDs and store in dictionary
    article_title_tokens = {}
    for aid, toks in zip(aids, tokenized_titles):
        tok_ids = [vocab[t] for t in toks if t in vocab]
        article_title_tokens[aid] = tok_ids
        
    del articles_df, titles, tokenized_titles # Free memory
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Processing behaviors to generate BM25 predictions...")
    t0_loop = time.time()
    processed = 0
    
    behaviors_path = MIND_LARGE_DIR / "behaviors.tsv"
    
    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        with open(behaviors_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                # behaviors.tsv format: [0]impression_id [1]user_id [2]time [3]history [4]inview
                imp_id = parts[0]
                history_str = parts[3]
                inview_str = parts[4]
                
                # Format history IDs
                if history_str.strip():
                    history = ["mind_" + x for x in history_str.split()]
                else:
                    history = []
                    
                # Format candidates
                candidates = ["mind_" + x for x in inview_str.split()]
                
                # Build BM25 query tokens from history titles
                query_ids = []
                for h in history[-50:]: # cap history to 50 for speed and relevance
                    if h in article_title_tokens:
                        query_ids.extend(article_title_tokens[h])
                
                if query_ids:
                    # BM25s can score a specific list of token IDs against all docs very fast
                    try:
                        scores = retriever.get_scores_from_ids(query_ids)
                        
                        cand_scores = []
                        for c in candidates:
                            if c in id_to_idx:
                                cand_scores.append(scores[id_to_idx[c]])
                            else:
                                cand_scores.append(-999.0)
                    except ValueError:
                        # Fallback if max token ID exceeds index
                        cand_scores = [-999.0] * len(candidates)
                else:
                    # Cold user
                    cand_scores = [-999.0] * len(candidates)
                    
                # Rank descending
                sorted_indices = np.argsort(cand_scores)[::-1]
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
