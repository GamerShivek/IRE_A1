# IRE Assignment 1 — News Recommendation Retrieval Pipeline

This repository implements a complete end-to-end information retrieval pipeline for news recommendation, evaluating both Lexical (BM25) and Semantic (Dense Embeddings) retrieval strategies on two major datasets: **MIND (Microsoft News)** and **EB-NeRD (Ekstra Bladet)**.

## Quick Start

```bash
# 1. Build the complete data pipeline (Parses raw data, splits chronologically, builds feature store)
make data
# Alternatively: python build_pipeline.py

# 2. Tune and run BM25 Lexical Retrieval
python tune_bm25.py
python run_bm25.py --tuned

# 3. Run Semantic Retrieval (Computes/loads embeddings and runs ANN search)
python run_embeddings.py

# 4. Evaluate all results (Computes Recall, MRR, nDCG, Diversity, Novelty + CI slicing)
python run_eval.py
```

## Pipeline Architecture

The project is structured sequentially to answer the assignment questions (Q1 to Q5).

### Q1: Reproducible Data Pipeline (`build_pipeline.py`)
Reads raw dataset files and converts them into a unified `.parquet` schema.
* **Temporal Split:** Rigorously splits logs chronologically (Train/Val/Test) without leaking future clicks into user history.
* **Feature Store:** Outputs ready-to-use `articles_features` and `user_features`.
* **Anti-Leakage Tests:** Runs assertions (`src/tests/test_no_leakage.py`) to guarantee chronological integrity.

### Q2: Lexical Retrieval (`run_bm25.py`, `tune_bm25.py`)
Implements keyword-based search.
* **Tokenization:** Cleans titles/abstracts and removes NLTK stopwords.
* **Indexing (`bm25s`):** Builds a highly optimized sparse-matrix index. Implements "Pseudo-BM25F" by duplicating title tokens to heavily weight title matches over abstracts.
* **Querying:** Bounding history to the 20 most recent tokens to maintain score variance.

### Q3: Semantic Retrieval (`run_embeddings.py`)
Implements meaning-based dense retrieval.
* **MIND:** Generates text embeddings on the fly using `sentence-transformers` (`all-MiniLM-L6-v2`).
* **EB-NeRD:** Loads provided pre-computed `google_bert_base_multilingual_cased` (mBERT) or Word2Vec embeddings.
* **User Vectors:** Applies a Time-Decayed Weighted Average (Early Fusion) to the user's click history, representing recent clicks more heavily than older ones.
* **Search:** Performs L2-Normalized Brute Force dot-product similarity (Cosine Similarity) to rank candidates.

### Q4: Evaluation Harness (`run_eval.py`)
A comprehensive offline evaluation suite located in `src/eval/`.
* **Accuracy Metrics:** Recall@K, AUC, MRR, nDCG@5, nDCG@10.
* **Beyond-Accuracy:** Measures Intra-list Diversity, Novelty, and Catalog Coverage.
* **Slicing & CI:** Automatically computes Bootstrap 95% Confidence Intervals and slices performance by "Cold-start vs. Warm" users.

## Codabench Submissions

The repository contains optimized scripts strictly for generating Codabench leaderboard submissions on the massive Test sets. They do not train models; they perform memory-efficient inference.

* `generate_codalab_mind_large.py`: Standard Early-Fusion semantic retrieval for MIND Large.
* `generate_codalab_mind_large_late_fusion.py`: Semantic retrieval using **Late Fusion** (comparing candidates to every history article individually and averaging top 3 matches).
* `generate_codalab_mind_large_bm25.py`: Lexical retrieval submission.
* `generate_codalab_ebnerd.py`: Semantic retrieval for EB-NeRD Testset, aggressively optimized with `float16` casting and chunk-streaming to prevent OOM errors.
* `generate_codalab_ebnerd_late_fusion.py`: Late Fusion implementation for EB-NeRD mBERT embeddings.

## Directory Layout

```text
project/
├── data/                      # Raw, interim, and processed parquet files
├── predictions/               # Generated Codabench .zip submission files
├── results/                   # Local validation metrics (JSON/Parquet)
├── src/                       
│   ├── eval/                  # Evaluation metrics (MRR, nDCG, Recall, etc.)
│   ├── parsers/               # MIND and EB-NeRD raw-to-unified schema parsers
│   ├── retrieval/             # BM25, Tokenization, ANN index, User Vectors
│   └── tests/                 # Anti-leakage assertions
├── build_pipeline.py          # Orchestrates Q1
├── run_bm25.py                # Orchestrates Q2
├── run_embeddings.py          # Orchestrates Q3
├── run_eval.py                # Orchestrates Q4
└── generate_codalab_*.py      # Codabench inference scripts (Q5)
```
