# IRE Assignment 1 — Reproducible Data Pipeline

## Q1: Reproducible Data Pipeline (MIND + EB-NeRD)

One command builds everything from raw files to a ready-to-use feature store.

---

## Quick Start

```bash
# From the project/ directory:
python build_pipeline.py          # Full pipeline — both datasets
python build_pipeline.py --mind-only
python build_pipeline.py --ebnerd-only
python build_pipeline.py --ebnerd-dataset small   # use ebnerd_small instead of demo

# Or via Make:
make data
make data-mind
make data-ebnerd
make test          # anti-leakage checks only (after pipeline has run)
make clean         # remove interim + processed (keeps raw files)
```

---

## Directory Layout

```
project/
├── data/
│   ├── raw/                      # symlinks / copies of raw downloaded data
│   ├── interim/
│   │   ├── mind/                 # unified schema, pre-split
│   │   └── ebnerd/               # unified schema, pre-split
│   └── processed/
│       ├── split_manifest.json   # cutoff dates + row counts (reproducibility)
│       ├── mind/{train,val,test}/
│       └── ebnerd/{train,val,test}/
├── src/
│   ├── parsers/
│   │   ├── mind_parser.py
│   │   └── ebnerd_parser.py
│   ├── temporal_split.py
│   ├── feature_store.py
│   └── tests/
│       └── test_no_leakage.py
├── build_pipeline.py             # one-command entry point
├── Makefile
└── README.md
```

Raw datasets are expected one level up (`../MINDsmall_train/`, `../ebnerd_demo/`, etc.) — exactly where the unzipped files land in the `A1/` directory.

---

## Pipeline Steps

### Step 1 — Check raw files
Fails fast with a clear wget message if any required file is missing. Never auto-downloads.

### Step 2 — Parse raw → interim unified schema
Both parsers output the **same three tables** regardless of source format:

| Table | Columns |
|---|---|
| `articles` | article_id (prefixed), title, abstract, body, category, published_time |
| `impressions` | impression_id, user_id, impression_time, candidate_article_ids, clicked_article_ids |
| `click_history` | user_id, article_id, click_time |

All IDs are prefixed (`mind_N123`, `ebnerd_9778745`) to prevent collisions if the datasets are ever joined.

### Step 3 — Temporal split (leakage-free)
- **Rule:** Sort by `impression_time`, cut by date. Never `shuffle=True`.
- MIND spans ~6 days → `test_days=1`, `val_days=1`.
- EB-NeRD demo spans ~14 days → `test_days=2`, `val_days=2`.
- `split_manifest.json` records all cutoff dates + row counts for reproducibility.

### Step 4 — Feature store
Per dataset per split:
- `articles_features.parquet` — article fields + placeholder `embedding` column (populated in Q3)
- `user_features.parquet` — history list + click_count (warm/cold user flag)

### Step 5 — Anti-leakage tests
Three assertions (see `src/tests/test_no_leakage.py`):
1. Temporal monotonicity: `max(train) < min(val) < min(test)`
2. History boundary: `max(history click_time) < min(impression_time)` for val/test
3. Per-impression spot-check (EB-NeRD only — MIND has no per-click timestamps)

---

## Known Dataset Limitations (Q6 design note)

| Dataset | Gap | Handling |
|---|---|---|
| MIND | No per-article `published_time` | Column is null; noted in schema |
| MIND | No article body text | `body` column is null; not fabricated |
| MIND | No per-click timestamps in history | `click_time` = impression_time of the impression containing the history. Limitation documented. |

These gaps affect Q3 (no body for BM25 fallback), Q4 (no recency decay for MIND history), and Q6's scale discussion.

---

## Reproducibility

Running the pipeline twice produces byte-identical output. The row-count summary printed at the end is diffable:

```
[mind] train=95070  val=31624  test=30271
[ebnerd] train=34760  val=7442  test=7878
```

The `split_manifest.json` records exact cutoff datetimes so any future rerun uses identical boundaries.
