"""
bm25_index.py
-------------
Build and persist a BM25 index over a dataset's full article catalog.

Uses bm25s (sparse-matrix backed) — orders of magnitude faster than rank_bm25
for large corpora (vectorised scoring via scipy sparse ops).

One index per dataset (not per split) — the article catalog is identical
across all splits; only impressions and user histories change.

bm25s API note:
  bm25s.tokenize() expects raw strings, not pre-tokenized lists.
  We bypass it by building the bm25s.tokenization.Tokenized namedtuple
  ourselves from our pre-tokenized lists and a shared vocabulary.
  This avoids double-tokenization and keeps our tokenizer (tokenize.py)
  as the single source of truth.
"""

import logging
import pickle
from pathlib import Path

import polars as pl
import bm25s
from bm25s.tokenization import Tokenized

from retrieval.tokenize import tokenize_batch

log = logging.getLogger(__name__)

DATASET_LANG = {"mind": "en", "ebnerd": "da"}
INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "indexes"


def _build_tokenized(token_lists: list[list[str]]) -> tuple[Tokenized, dict[str, int]]:
    """
    Build a bm25s Tokenized corpus from pre-tokenized lists.

    Returns (Tokenized, vocab) where vocab maps token → integer id.
    """
    vocab: dict[str, int] = {}
    ids: list[list[int]] = []

    for doc in token_lists:
        doc_ids = []
        for tok in doc:
            if tok not in vocab:
                vocab[tok] = len(vocab)
            doc_ids.append(vocab[tok])
        ids.append(doc_ids)

    return Tokenized(ids=ids, vocab=vocab), vocab


def _queries_to_tokenized(
    query_token_lists: list[list[str]],
    vocab: dict[str, int],
) -> Tokenized:
    """
    Convert pre-tokenized query lists to a bm25s Tokenized object
    using the SAME vocabulary as the index. Unknown query tokens are
    silently dropped (they wouldn't score against any document anyway).
    """
    ids = [
        [vocab[tok] for tok in q if tok in vocab]
        for q in query_token_lists
    ]
    return Tokenized(ids=ids, vocab=vocab)


def _articles_text(
    articles: pl.DataFrame,
    dataset: str,
    title_weight: int = 2,
) -> tuple[list[str], list[list[str]]]:
    """
    Return (article_ids, tokenized_docs) for BM25 indexing.

    Text indexed:
      MIND   : title + abstract  (no body — Q1 finding)
      EB-NeRD: title + abstract  (abstract = subtitle in Q1 schema;
                                  body excluded for fair lexical/semantic
                                  comparison — noted in design note)

    Title-weighting (BM25F-style approximation):
      Title tokens are repeated `title_weight` times before abstract tokens.
      This inflates TF for title terms, making them dominate scoring.
      bm25s has no native multi-field support, so token repetition is the
      standard workaround at this scale.
    """
    lang = DATASET_LANG[dataset]
    article_ids = articles["article_id"].to_list()

    tokenized = []
    for row in articles.iter_rows(named=True):
        title    = row.get("title")    or ""
        abstract = row.get("abstract") or ""
        title_toks    = tokenize_batch([title],    lang=lang)[0]
        abstract_toks = tokenize_batch([abstract], lang=lang)[0]
        # Repeat title tokens to boost their TF weight
        doc_toks = title_toks * title_weight + abstract_toks
        tokenized.append(doc_toks)

    return article_ids, tokenized


def build_index(
    articles: pl.DataFrame,
    dataset: str,
    force_rebuild: bool = False,
    k1: float = 1.5,
    b: float = 0.75,
    title_weight: int = 2,
) -> tuple[bm25s.BM25, list[str], dict[str, int], dict[str, int]]:
    """
    Build (or load from cache) a bm25s index for `dataset`.

    Args:
        k1           : BM25 term-frequency saturation (default 1.5)
        b            : BM25 length normalisation (default 0.75)
        title_weight : repeat title tokens N times before abstract tokens
                       (crude BM25F approximation; 1 = equal weight)

    Cache key includes k1, b, title_weight — different hyperparams get
    different cache files so switching doesn't invalidate the default cache.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # Include hyperparams in cache filename so different configs don't collide
    cache_name = f"bm25s_{dataset}_k1{k1}_b{b}_tw{title_weight}.pkl"
    cache_path = INDEX_DIR / cache_name

    if cache_path.exists() and not force_rebuild:
        log.info("[%s] Loading cached bm25s index from %s", dataset, cache_path)
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["retriever"], data["article_ids"], data["id_to_idx"], data["vocab"]

    log.info(
        "[%s] Building bm25s index: k1=%.2f  b=%.2f  title_weight=%d  articles=%d",
        dataset, k1, b, title_weight, len(articles),
    )
    article_ids, tokenized = _articles_text(articles, dataset, title_weight=title_weight)

    # Filter empty docs
    non_empty = [(aid, toks) for aid, toks in zip(article_ids, tokenized) if toks]
    n_empty = len(article_ids) - len(non_empty)
    if n_empty:
        log.warning("  %d articles had empty token lists — excluded", n_empty)

    final_ids  = [x[0] for x in non_empty]
    final_toks = [x[1] for x in non_empty]

    corpus_tokenized, vocab = _build_tokenized(final_toks)

    retriever = bm25s.BM25(k1=k1, b=b, method="bm25+")
    retriever.index(corpus_tokenized, show_progress=False)

    id_to_idx = {aid: i for i, aid in enumerate(final_ids)}

    data = {
        "retriever":   retriever,
        "article_ids": final_ids,
        "id_to_idx":   id_to_idx,
        "vocab":       vocab,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(
        "  Index built → %s  (%d docs, vocab=%d)", cache_path, len(final_ids), len(vocab)
    )

    return retriever, final_ids, id_to_idx, vocab


def load_or_build_index(
    processed_dir: Path,
    dataset: str,
    force_rebuild: bool = False,
    k1: float = 1.5,
    b: float = 0.75,
    title_weight: int = 2,
) -> tuple[bm25s.BM25, list[str], dict[str, int], dict[str, int]]:
    """Load articles from train split (full catalog) and build/load the index."""
    articles_path = processed_dir / dataset / "train" / "articles_features.parquet"
    if not articles_path.exists():
        raise FileNotFoundError(
            f"articles_features.parquet not found at {articles_path}\n"
            "Run build_pipeline.py first."
        )
    articles = pl.read_parquet(articles_path)
    return build_index(articles, dataset, force_rebuild=force_rebuild,
                       k1=k1, b=b, title_weight=title_weight)
