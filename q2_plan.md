# Q2 Plan: Lexical Candidate Generation (BM25)

**Goal:** for each user, build a text query from their click history, retrieve top-K candidate articles via BM25 over `title + abstract`, and report recall@K for K ∈ {50, 100, 200} — for both MIND and EB-NeRD, on val and test splits.

Builds directly on Q1's feature store (`articles_features.parquet`, `user_features.parquet` per split) — nothing here should touch raw files again.

---

## 0. Where this fits in the repo

```
src/
├── retrieval/
│   ├── bm25_index.py       # build inverted index per dataset/split
│   ├── bm25_query.py       # user history → query string
│   ├── bm25_retrieve.py    # retrieve top-K per user
│   └── tokenize.py         # dataset-aware tokenization
├── eval/
│   └── recall_at_k.py      # shared by Q2 and Q3
└── run_bm25.py             # orchestrates: build index → retrieve → eval → save results
```

Keep `recall_at_k.py` dataset-and-method-agnostic now — Q3 (embeddings) will reuse it, and Q4's harness will too.

---

## 1. What text to index — per dataset

| dataset | index over | notes |
|---|---|---|
| MIND | `title + abstract` | no `body` field exists (Q1 finding) — don't leave a blank body concatenated in, that would just add noise/nulls |
| EB-NeRD | `title + subtitle` | assignment explicitly says titles/abstracts only for Q2, even though EB-NeRD also has `body` — stick to title+abstract for a fair lexical/semantic comparison later; note in design note that body is available but intentionally excluded here |

Build one BM25 index **per dataset**, indexed over that dataset's full article catalog (65,238 for MIND, 11,777 for EB-NeRD) — not per-split. The article catalog doesn't change across splits; only which impressions/users you *evaluate against* changes. Indexing per-split would be wasteful and wrong (test articles should already be discoverable, they're just not yet in anyone's history).

---

## 2. Tokenization — don't reuse one pipeline blindly

- **MIND (English):** lowercase, strip punctuation, standard English stopword list, optional stemming (Porter/Snowball). Off-the-shelf NLTK/spaCy English tokenizer is fine.
- **EB-NeRD (Danish):** lowercase, strip punctuation, **Danish stopword list** (not English — reusing an English stopword list here silently degrades BM25 since Danish function words won't be filtered). Use spaCy's `da_core_news_sm` or NLTK's Danish stopwords. Danish compounding (long compound nouns) may also reduce raw term overlap — worth noting as a limitation, not solving from scratch.
- Write `tokenize.py` as `tokenize(text: str, lang: str) -> list[str]` so both parsers call the same function with a `lang` flag — keeps the dataset-agnostic downstream code truly agnostic.

**Action item:** sanity-check tokenizer output on 5 real EB-NeRD titles before running the full index build — a silently-broken Danish tokenizer (e.g. falling back to English rules) will tank BM25 recall without throwing any error.

---

## 3. Building the BM25 index

- Library: `rank_bm25` (pure Python, fine at this scale — 65K/12K docs) or `bm25s` (faster, drop-in if `rank_bm25` is too slow on MIND's 65K articles). Don't hand-roll unless you have a specific reason to.
- Input: list of tokenized `title+abstract` strings, one per article, indexed by `article_id`.
- Output: a fitted BM25 object + an `article_id` ↔ index-position mapping (needed to map BM25's scored positions back to real article IDs).
- Persist the index per dataset (pickle or rebuild-on-demand) so `run_bm25.py` doesn't refit on every run — refitting a 65K-doc index each time is wasted work if you're only changing K or the eval slice downstream.

---

## 4. Query construction — from user history

Given a `user_id` and a split (val/test), pull their **bounded** click history from Q1's `user_features.parquet` for that split (already correctly leakage-filtered per Q1's Step 5 checks).

- Query text = concatenation of the **titles** of their historical clicked articles (assignment's suggested approach). Cap history length (e.g. last 20–50 clicks) if a user has an unusually long history — otherwise the query balloons and BM25 term weighting gets diluted.
- Tokenize the query with the *same* `tokenize()` function and `lang` flag as the index.

**Cold users (MIND has ~10K–13K of these per split; EB-NeRD has none — Q1 finding):**
- A cold user has an empty or near-empty history → empty query → BM25 has nothing to score against.
- Fallback options (pick one, document the choice):
  - (a) Skip cold users entirely from Q2's recall@K computation, report their count as a known gap.
  - (b) Fallback query = the user's most recent single click if any exists, or the article's `category` field.
  - (c) Fallback = most-popular-articles-in-training as candidates (== a "popularity baseline" — cold users are hard for *any* content-based method, this is expected and worth stating).
- Whichever you pick, this is exactly the kind of decision Q4's cold-vs-warm slice will surface as a story — MIND lexical retrieval will visibly struggle on cold users, and that's a legitimate, expected finding, not a bug to hide.

---

## 5. Retrieval

For each user (with a non-empty query) in val/test:
1. Score all articles in the dataset's BM25 index against the query.
2. Take top-K article IDs for K ∈ {50, 100, 200}.
3. Store `(user_id, impression_id, K, retrieved_article_ids)` — keep per-K results, don't just store top-200 and slice later unless you're certain top-200 ⊇ top-100 ⊇ top-50 ordering is preserved (it should be, but store explicitly to avoid a subtle bug).

**Important:** retrieve per (user, impression) pair, not just per user — a user might have multiple impressions in a split, each with its own candidate/clicked-article ground truth, and the assignment's recall@K is computed against the impression's actual clicked articles.

---

## 6. Recall@K — what "ground truth" means here

For a given impression:
- Ground truth = the `clicked_article_ids` for that impression (from Q1's `impressions` table).
- `recall@K = |retrieved_top_K ∩ clicked_article_ids| / |clicked_article_ids|`
- Average this across all impressions (with a non-cold-fallback query) in val/test, separately for each K.

Two subtleties to get right:
1. **Some impressions have zero clicks** (if EB-NeRD/MIND behaviors include shown-but-nothing-clicked impressions) — decide whether these contribute a 0 to the average or are excluded entirely, and be consistent. Excluding them is more standard (recall is undefined with an empty ground-truth set).
2. Report recall@K **per split** (val and test separately) and **per dataset** — don't average MIND and EB-NeRD together, they're different tasks/languages.

---

## 7. Output structure

```
results/bm25/
├── mind/
│   ├── val/recall_at_k.json     # {50: 0.xx, 100: 0.xx, 200: 0.xx}
│   └── test/recall_at_k.json
└── ebnerd/
    ├── val/recall_at_k.json
    └── test/recall_at_k.json
```
Plus raw retrieved-candidates files if Q4's harness needs to re-consume them (it likely will, per the assignment: "run your evaluation harness on both BM25 and embedding-based retrieval results").

---

## Checklist

- [ ] `tokenize.py` with English + Danish paths, sanity-checked on real samples
- [ ] BM25 index built once per dataset over full article catalog (title+abstract only)
- [ ] Query builder pulling from Q1's leakage-safe `user_features.parquet`
- [ ] Explicit, documented cold-user fallback strategy (not silently dropped)
- [ ] Top-K retrieval stored per (user, impression) for K ∈ {50, 100, 200}
- [ ] `recall_at_k.py` — dataset-agnostic, reusable for Q3
- [ ] Recall@K computed and saved per dataset × split × K
- [ ] Note in design note: EB-NeRD body excluded intentionally, Danish tokenization caveat, MIND cold-start fallback choice and its effect on recall