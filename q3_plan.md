# Q3 Plan: Semantic Candidate Generation (Embeddings)

**Goal:** for each user, build a semantic query vector from their click history, retrieve top-K candidates via nearest-neighbor search over article embeddings, report recall@K for K ∈ {50, 100, 200}, and compare against Q2's BM25 results — same datasets, same splits, same `recall_at_k.py`.

Reuses Q1's feature store and Q2's evaluation harness directly. Only the retrieval mechanism changes.

---

## 0. Where this fits in the repo

```
src/
├── retrieval/
│   ├── embeddings_load.py    # load provided embeddings / compute your own
│   ├── ann_index.py          # build FAISS (or brute-force) index per dataset
│   ├── user_vector.py        # user history → embedding vector
│   └── embed_retrieve.py     # retrieve top-K per user
├── eval/
│   └── recall_at_k.py        # UNCHANGED from Q2 — reused as-is
└── run_embeddings.py         # orchestrates: load/build embeddings → index → retrieve → eval
```

Nothing in `eval/` should change for Q3 — if you find yourself editing `recall_at_k.py` to make embeddings "work," that's a sign of a bug in the retrieval code, not a reason to adjust the metric.

---

## 1. Embeddings — provided vs. computed, per dataset

| dataset | option | plan |
|---|---|---|
| EB-NeRD | provided | Use `Ekstra_Bladet_word2vec.zip` or `google_bert_base_multilingual_cased.zip` (already downloaded in Part 0). Start with word2vec — cheaper, faster, lets you get the retrieval pipeline correct before adding a heavier BERT pass. |
| EB-NeRD | computed (stretch) | If time allows, run multilingual BERT over title+abstract yourself and compare against the provided word2vec vectors — gives you a legitimate "does contextual embedding beat static embedding" ablation for the design note. |
| MIND | provided | **None ships with MIND** — must compute your own. |
| MIND | computed | Use a pretrained sentence embedding model (e.g. `sentence-transformers/all-MiniLM-L6-v2`) over `title + abstract` (no body, per Q1/Q2 finding). Don't fine-tune — out-of-the-box embeddings are the right scope for this assignment; fine-tuning is a Q3-stretch or later-assignment concern, not baseline scope. |

**Action item:** confirm the provided EB-NeRD embeddings are keyed by `article_id` and cover the full 11,777-article catalog before building anything on top — a partial-coverage embedding file (e.g. only covering articles in the original impression logs, not the demo bundle's catalog) would silently produce nulls for some articles.

---

## 2. Building the ANN index

- **Brute-force cosine similarity is fine at this scale** — 11,777 (EB-NeRD) and 65,238 (MIND) articles both fit comfortably in memory as dense matrices; a full similarity matrix per query is cheap. Don't reach for FAISS unless brute-force retrieval is actually too slow in practice — premature complexity here just adds a place for id-mapping bugs to hide (the exact bug class flagged as a suspect in Q2's EB-NeRD near-random investigation).
- If you do use FAISS: start with a flat index (`IndexFlatIP` on L2-normalized vectors, i.e. exact cosine) before any approximate/quantized index — exactness matters more than speed here, and approximate indexes trade recall for speed you don't need at this scale.
- **Normalize all article vectors to unit length** before indexing (whether brute-force or FAISS) so dot product = cosine similarity — an easy thing to forget that silently changes what "nearest" means.
- Build one index per dataset over the full article catalog, same reasoning as Q2 (catalog doesn't change across splits, only which impressions you evaluate).

---

## 3. User vector construction — the semantic analogue of Q2's query

Given a user's bounded click history (same `user_features.parquet` per split, same leakage guarantees Q1 verified):

- **Baseline: mean-pool the embeddings of their clicked articles.**
- Apply the **same recency-capping lesson learned in Q2**: don't naively average every article a user has ever clicked. A 147-article mean-pool for EB-NeRD users will wash out signal the same way an overlong BM25 query collapsed IDF — averaging in unrelated topics dilutes the vector toward the corpus centroid. Cap history length (same window as tuned in Q2, or re-tune independently for embeddings — averaging behaves differently than term-frequency scoring, so the same magic number isn't guaranteed to transfer).
- **Stretch, worth trying given the time budget**: recency-weighted mean (more recent clicks weighted higher) rather than a flat average — directly extends the recency-weighting idea flagged as a Q2 follow-up, and is a natural fit for embeddings since it's just a weighted sum.
- **Cold users:** same problem as Q2, same policy — no history means no vector to pool. Reuse whatever fallback strategy you picked and documented in Q2 (skip + report count, or a fallback) for consistency across the two methods; don't invent a different cold-user policy for Q3 that makes the two results incomparable.

---

## 4. Retrieval

For each (user, impression) pair with a valid history vector:
1. Cosine-similarity search of the user vector against the full article embedding matrix.
2. Take top-K article IDs for K ∈ {50, 100, 200}.
3. Store `(user_id, impression_id, K, retrieved_article_ids)` in the same shape as Q2's `candidates.parquet` — this is what lets `recall_at_k.py` run unmodified over both methods' outputs.

---

## 5. Evaluation — reuse, don't reinvent

- Run the **exact same** `recall_at_k.py` from Q2 over the embedding-based candidates.
- Compute the **same random-baseline and multiple-over-random fields** you added to Q2's output — this is what makes the Q2-vs-Q3 comparison in Q6's design note legitimate rather than apples-to-oranges.
- Report side-by-side, per dataset × split × K: BM25 recall@K vs. embedding recall@K vs. random baseline.

**Expected result to sanity-check against:** based on Q2's finding that BM25's core weakness is vocabulary mismatch (different words, same topic), embeddings should show a real, visible lift over BM25's multiple-over-random — especially for MIND's larger catalog and more paraphrased/varied journalism. If embeddings come out *at or below* BM25's numbers, don't write that up as "surprisingly, lexical beats semantic" without first checking for the same bug classes that hit Q2's first EB-NeRD run (silently degenerate vectors, misaligned article_id↔row-index mapping, or an over-long/diluted history vector) — those exact bug shapes tend to reappear whenever a "should clearly help" method comes out flat.

---

## 6. Q3's own specific deliverable: lexical vs. semantic comparison

The assignment explicitly asks (Q3.5): "Compare lexical vs. semantic retrieval: which works better? On which slices?" Plan for this now, not as an afterthought:

- Reuse Q1's warm/cold slice (remember: EB-NeRD has zero cold users, so this comparison is MIND-only for the cold-start axis — same limitation as Q4).
- Add a second slice if time allows: head (popular) vs. tail (rare) articles — a plausible hypothesis worth testing is that BM25 does relatively better on tail articles (exact rare-term matches are BM25's strength) while embeddings do relatively better on head articles (more training signal for the embedding space, more paraphrase variety to bridge). Test it rather than assume it.
- Write the comparison as a table (dataset × split × method × slice), not just prose — this maps directly into Q6's "observations from experiments" requirement.

---

## Checklist

- [ ] Confirm EB-NeRD provided embeddings cover full article catalog (no silent nulls)
- [ ] Compute MIND embeddings (title+abstract, off-the-shelf sentence embedding model, no fine-tuning)
- [ ] Brute-force cosine (or FAISS flat) index per dataset, vectors L2-normalized
- [ ] User vector = capped, mean-pooled (or recency-weighted) history embeddings
- [ ] Cold-user policy matches Q2's exactly, for a fair comparison
- [ ] Candidates stored in the same schema as Q2's `candidates.parquet`
- [ ] `recall_at_k.py` reused unmodified; random-baseline fields included
- [ ] Side-by-side BM25 vs. embedding table, per dataset × split × K
- [ ] Warm/cold and (if time) head/tail slice comparison for Q3.5 and Q4
- [ ] If embeddings underperform BM25 anywhere, rule out bugs (id-mapping, degenerate vectors, over-long history) before concluding it's a genuine finding