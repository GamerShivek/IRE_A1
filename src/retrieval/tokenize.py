"""
tokenize.py
-----------
Dataset-aware tokenization.

Usage:
    tokens = tokenize("Some article title", lang="en")
    tokens = tokenize("Dansk overskrift tekst", lang="da")

Design decisions:
- Regex word splitting (no punkt dependency — punkt download is slow and unreliable).
- NLTK stopword lists for English and Danish — only stopwords from nltk, not punkt.
- No stemming by default to keep terms interpretable; Porter stemmer is available
  via stem=True flag if you want to experiment.
- Danish compound nouns (e.g. "statsminister") are NOT split — this is a known
  limitation noted in the design note. Compound splitting requires a Danish-specific
  dictionary and is not worth implementing from scratch for this assignment.
"""

import re
import logging
from functools import lru_cache

log = logging.getLogger(__name__)

# ── Stopword loading ────────────────────────────────────────────────────────
def _load_stopwords(lang: str) -> frozenset:
    """Load NLTK stopwords; fall back to a minimal hardcoded set if unavailable."""
    nltk_lang = {"en": "english", "da": "danish"}.get(lang, "english")
    try:
        from nltk.corpus import stopwords
        return frozenset(stopwords.words(nltk_lang))
    except Exception:
        log.warning(
            "NLTK stopwords for '%s' not available — using minimal fallback list.", lang
        )
        _fallback = {
            "en": frozenset({"a", "an", "the", "is", "in", "on", "at", "to", "of",
                              "and", "or", "for", "with", "this", "that", "it",
                              "as", "by", "be", "was", "are", "were"}),
            "da": frozenset({"og", "i", "det", "at", "en", "er", "den", "til",
                              "af", "de", "med", "vi", "for", "ikke", "der",
                              "om", "et", "på", "som", "han"}),
        }
        return _fallback.get(lang, frozenset())


@lru_cache(maxsize=4)
def _stopwords_cached(lang: str) -> frozenset:
    return _load_stopwords(lang)


# ── Core tokenizer ──────────────────────────────────────────────────────────
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # Unicode letters only, no digits


def tokenize(
    text: str | None,
    lang: str = "en",
    stem: bool = False,
    min_len: int = 2,
    max_history_words: int | None = None,
) -> list[str]:
    """
    Tokenize `text` for BM25 indexing or query construction.

    Args:
        text    : raw text string (may be None or empty → returns [])
        lang    : 'en' for English (MIND), 'da' for Danish (EB-NeRD)
        stem    : if True, apply Porter stemmer (English only)
        min_len : drop tokens shorter than this
        max_history_words : if set, truncate output to this many tokens
                            (used to cap ballooning query sizes)

    Returns: list of lowercase tokens with stopwords removed.
    """
    if not text:
        return []

    tokens = _WORD_RE.findall(text.lower())
    stopwords = _stopwords_cached(lang)
    tokens = [t for t in tokens if t not in stopwords and len(t) >= min_len]

    if stem and lang == "en":
        try:
            from nltk.stem import PorterStemmer
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(t) for t in tokens]
        except Exception:
            pass  # stemmer unavailable — proceed without

    if max_history_words is not None:
        tokens = tokens[:max_history_words]

    return tokens


def tokenize_batch(
    texts: list[str | None],
    lang: str = "en",
    stem: bool = False,
) -> list[list[str]]:
    """Tokenize a list of texts. Convenience wrapper."""
    return [tokenize(t, lang=lang, stem=stem) for t in texts]


# ── Sanity check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import polars as pl
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    base = Path(__file__).resolve().parents[3]

    print("=== English (MIND) tokenizer sanity check ===")
    mind_art = pl.read_parquet(
        base / "data/processed/mind/train/articles_features.parquet"
    )
    for row in mind_art.head(5).iter_rows(named=True):
        text = f"{row['title'] or ''} {row['abstract'] or ''}"
        tokens = tokenize(text, lang="en")
        print(f"  {text[:60]!r}  →  {tokens[:8]}")

    print("\n=== Danish (EB-NeRD) tokenizer sanity check ===")
    ebnerd_art = pl.read_parquet(
        base / "data/processed/ebnerd/train/articles_features.parquet"
    )
    for row in ebnerd_art.head(5).iter_rows(named=True):
        text = f"{row['title'] or ''} {row['abstract'] or ''}"
        tokens = tokenize(text, lang="da")
        print(f"  {text[:60]!r}  →  {tokens[:8]}")
