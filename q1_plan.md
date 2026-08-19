# Q1 Plan: Reproducible Data Pipeline (MIND + EB-NeRD)

**Goal:** One command (`make data` or `python build_pipeline.py`) that takes both datasets from raw downloaded files to a ready-to-use feature store: clean unified tables, a leakage-free temporal split, and per-article/per-user features.

---

## 0. Repo layout

```
project/
├── data/
│   ├── raw/
│   │   ├── mind/            # unzipped MINDsmall_train/, MINDsmall_dev/
│   │   └── ebnerd/          # unzipped ebnerd_demo/, ebnerd_small/
│   ├── interim/             # unified schema, pre-split
│   └── processed/           # post-split, feature store
│       ├── mind/{train,val,test}/
│       └── ebnerd/{train,val,test}/
├── src/
│   ├── parsers/
│   │   ├── mind_parser.py
│   │   └── ebnerd_parser.py
│   ├── unify_schema.py
│   ├── temporal_split.py
│   ├── feature_store.py
│   └── tests/
│       └── test_no_leakage.py
├── build_pipeline.py        # or Makefile
└── README.md
```

Keep `parsers/` dataset-specific and everything downstream (`unify_schema.py` onward) dataset-agnostic — that's the whole point of Step 2.

---

## 1. Raw data → understand what you actually have

### MIND-small (English, TSV)
| File | Columns |
|---|---|
| `news.tsv` | News ID, Category, SubCategory, Title, Abstract, URL, Title Entities, Abstract Entities |
| `behaviors.tsv` | Impression ID, User ID, Time, History (space-sep News IDs), Impressions (space-sep `NewsID-Label` pairs, e.g. `N123-1 N234-0`) |

No `body` field — only title + abstract. No separate "history" file; history is embedded per-impression in `behaviors.tsv` (same list repeats across a user's impressions).

### EB-NeRD (Danish, Parquet)
| File | Columns (approx.) |
|---|---|
| `articles.parquet` | article_id, title, subtitle, body, category, topics, published_time, ner_clusters, ... |
| `behaviors.parquet` | impression_id, user_id, impression_time, session_id, article_ids_inview, article_ids_clicked, ... |
| `history.parquet` | user_id, article_id_fixed (list), impression_time_fixed (list), scroll_percentage_fixed, read_time_fixed |

Here, history is a **separate table**, not embedded per-impression — different shape from MIND.

**Action item:** before writing any parsing code, `df.columns` / `pl.read_parquet(...).schema` and inspect 2-3 rows of each raw file yourself. Exact column names can drift slightly across dataset versions — don't hardcode from memory (mine or otherwise) without checking.

**Use the starter code for EB-NeRD here.** `pip install .` the `ebrec` package from `ebnerd-benchmark`, and reuse its loading utilities (demonstrated in `examples/datasets/ebnerd_overview.ipynb`) to load `articles.parquet` / `behaviors.parquet` / `history.parquet` and join each user's click history onto their impressions with binary click labels. This does most of the heavy lifting for `ebnerd_parser.py` below — don't rewrite the polars join logic from scratch. There is no equivalent starter code for MIND, so `mind_parser.py` is written from zero.

---

## 2. Unified schema — the target for both parsers

Design three tables that both `mind_parser.py` and `ebnerd_parser.py` must output, regardless of source format:

**`articles`**
| column | type | notes |
|---|---|---|
| `article_id` | str | prefix with dataset name if IDs could collide later, e.g. `mind_N123` |
| `title` | str | |
| `abstract` | str | EB-NeRD: use `subtitle`; MIND: use `Abstract` |
| `body` | str \| null | MIND has none — leave null, don't fabricate |
| `category` | str | |
| `published_time` | datetime | MIND has no timestamp per article — leave null, flag this as a known dataset gap in your design note |

**`impressions`** (one row per impression = one shown list)
| column | type | notes |
|---|---|---|
| `impression_id` | str | |
| `user_id` | str | |
| `impression_time` | datetime | **this is what you sort on for the temporal split** |
| `candidate_article_ids` | list[str] | all articles shown |
| `clicked_article_ids` | list[str] | subset that were clicked |

**`click_history`** (one row per user, or per user-event — pick one and be consistent)
| column | type | notes |
|---|---|---|
| `user_id` | str | |
| `article_id` | str | |
| `click_time` | datetime | for MIND this is just "before this impression's Time" since MIND doesn't timestamp individual history clicks — note this limitation |

**Decision to make explicitly and write down:** how to handle MIND's lack of per-click timestamps and article body text. Don't silently drop the fields — state the limitation in your design note, since Q6 asks for exactly this kind of observation.

---

## 3. Cleaning / parsing steps (per dataset)

- **MIND:** split `Impressions` column on whitespace, then split each token on `-` into `(article_id, label)`. Split `History` on whitespace into a list. Handle empty `History` (cold-start users) as an empty list, not null/crash. Written from scratch — no starter code available.
- **EB-NeRD:** `article_ids_inview` / `article_ids_clicked` are likely already list-typed in the parquet (polars handles nested lists natively) — verify, don't assume you need to parse strings. Start from `ebrec`'s load/join output (see Step 1) rather than reading the raw parquet files directly — you're reshaping its output into the unified schema, not parsing from zero.
- Both: drop or flag rows with missing `user_id`/`impression_time`; log how many rows dropped (reproducibility — this number should be stable across reruns).
- Both: dedupe exact-duplicate impression rows if any exist.

---

## 4. Temporal split — the part most people get wrong

**Rule: sort by `impression_time`, then cut by date. Never `train_test_split` with `shuffle=True`.**

Concrete plan:
1. Find `min(impression_time)` and `max(impression_time)` per dataset.
2. Pick cutoffs as **last N days = test, preceding M days = val, rest = train** (mirrors what the assignment prescribes). Reasonable starting point: last 1 day = test, prior 1 day = val, rest = train — but check what date range each dataset actually spans first (MIND-small is famously short, ~6 days; EB-NeRD demo is a few weeks) and adjust N/M so each split isn't empty or absurdly tiny.
3. **A user's click history for an impression must only include clicks that happened strictly before that impression's time.** This is the leakage rule Q9 wants you to test for explicitly.
4. Write the cutoff dates and split sizes (row counts) to a small `split_manifest.json` — useful for your design note and for reproducibility.

---

## 5. Feature store

Keep it simple — this doesn't need to be a database, just organized, fast-to-load files:

- `data/processed/{dataset}/articles_features.parquet` — article_id + text fields + (later) embeddings once Q3 computes them.
- `data/processed/{dataset}/user_features.parquet` — user_id + click history (list of article_ids, up to split boundary) + click count (recency proxy).
- Everything keyed by `article_id` / `user_id` so BM25 (Q2) and embedding retrieval (Q3) can `join`/`lookup` without re-parsing raw files.

---

## 6. One-command rebuild

`build_pipeline.py` (or a `Makefile`) should run, in order:
1. Check raw files exist under `data/raw/{dataset}/` (fail with a clear message telling the user to `wget` if missing — don't auto-download inside the script unless you want to bake in the `wget` commands too).
2. Call `mind_parser.py` and `ebnerd_parser.py` → write unified tables to `data/interim/`.
3. Call `temporal_split.py` → write split tables + `split_manifest.json` to `data/processed/{dataset}/{train,val,test}/`.
4. Call `feature_store.py` → write `articles_features.parquet` / `user_features.parquet` per split.
5. Print a short summary (row counts per table per split) so a re-run's output is diffable.

Idempotency check: running the command twice should produce byte-identical output (or at least identical row counts) — this is what "reproducible" means for grading purposes.

---

## 7. Anti-leakage test (ties into Q9, do it now not later)

Write `test_no_leakage.py`:
- Assert every `click_history` entry used to build a user's feature vector for a given impression has `click_time < impression_time` of that impression.
- Assert no `impression_time` in the val/test split is `<=` the max `impression_time` in train.
- Run this as part of the pipeline (or CI) — a failing assertion here is exactly the kind of thing Q9 wants documented.

---

## Checklist

- [ ] Inspect raw file schemas for both datasets by hand
- [ ] `mind_parser.py` → unified schema
- [ ] `ebnerd_parser.py` → unified schema
- [ ] Explicit notes on MIND's missing body/timestamp fields
- [ ] `temporal_split.py` with sensible cutoffs per dataset's actual date range
- [ ] `split_manifest.json` output
- [ ] `feature_store.py` writing article + user feature tables
- [ ] `build_pipeline.py` / `Makefile` running all of the above in one command
- [ ] `test_no_leakage.py` passing
- [ ] Row-count summary printed/logged for reproducibility check