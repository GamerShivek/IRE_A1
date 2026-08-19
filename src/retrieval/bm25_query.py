"""
bm25_query.py
-------------
Build a BM25 query token list from a user's click history.

Cold-user strategy: skip cold users from recall@K (they have no history →
empty query → BM25 cannot score). Their count is logged and reported in the
results JSON. This keeps recall honest (not inflated by popular-fallback)
and makes the cold-start gap visible in Q4's slice analysis.

Query token budget (KEY DESIGN DECISION):
  BM25 IDF is highest for rare terms. A very long query floods the model with
  common terms that appear in many documents, collapsing score variance and
  producing near-random ranking.

  EB-NeRD users have median 147-article histories. With 30 articles × ~5
  tokens/title = 145-token query, score spread collapses from ~40 to ~16
  points across 11,777 docs — effectively random ranking.

  Fix: cap by TOKEN count (MAX_QUERY_TOKENS), accumulated from the most
  recent history articles backward. This keeps queries short and focused.
"""

import logging
from retrieval.tokenize import tokenize

log = logging.getLogger(__name__)

# Maximum tokens in the BM25 query — empirically tuned:
# cap=20 gives spread≈30 for EB-NeRD; cap=30 gives spread≈39.
# Higher caps hurt IDF discrimination on EB-NeRD's 11K-article catalog.
# MIND has 65K articles and typically shorter histories — less sensitive.
MAX_QUERY_TOKENS = 20

DATASET_LANG = {"mind": "en", "ebnerd": "da"}


def build_query_tokens(
    history_article_ids: list[str] | None,
    article_title_lookup: dict[str, str],
    dataset: str,
    max_history: int = 50,          # max articles to look back through
    max_tokens: int = MAX_QUERY_TOKENS,
) -> list[str]:
    """
    Build a BM25 query token list from a user's click history.

    Accumulates tokens from the most recent history articles backward until
    `max_tokens` is reached (token budget), or `max_history` articles are
    exhausted — whichever comes first.

    Args:
        history_article_ids  : article IDs from user_features (leakage-safe)
        article_title_lookup : dict {article_id → title}
        dataset              : 'mind' or 'ebnerd' (chooses tokenizer language)
        max_history          : how many articles to look back through at most
        max_tokens           : hard cap on total query token count

    Returns: list of tokens; empty list means cold user.
    """
    if not history_article_ids:
        return []

    lang = DATASET_LANG.get(dataset, "en")
    # Most recent articles are at the end — iterate in reverse
    recent = list(history_article_ids)[-max_history:][::-1]

    query_tokens: list[str] = []
    for aid in recent:
        title = article_title_lookup.get(aid)
        if not title:
            continue
        toks = tokenize(title, lang=lang)
        remaining = max_tokens - len(query_tokens)
        if remaining <= 0:
            break
        query_tokens.extend(toks[:remaining])

    return query_tokens


def build_article_title_lookup(articles_df) -> dict[str, str]:
    """Build a dict {article_id: title} from an articles DataFrame."""
    return dict(zip(
        articles_df["article_id"].to_list(),
        [(t or "") for t in articles_df["title"].to_list()],
    ))

