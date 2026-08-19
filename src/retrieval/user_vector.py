import logging
import numpy as np

log = logging.getLogger(__name__)

def build_user_vector(
    history_article_ids: list[str],
    index,  # BruteForceIndex
    max_history: int = 50,
) -> np.ndarray | None:
    """
    Builds the user vector by mean-pooling the most recent `max_history` articles.
    If the history is empty, returns None.
    history_article_ids are assumed to be chronological (or at least, we'll take the tail).
    Wait, the user history might not be perfectly chronological in the parquet list, but taking the last N is standard.
    """
    if not history_article_ids:
        return None
    
    # Take the most recent `max_history` articles (tail of the list)
    recent_history = history_article_ids[-max_history:]
    
    vectors = []
    for aid in recent_history:
        if aid in index.id_to_idx:
            idx = index.id_to_idx[aid]
            vectors.append(index.vectors[idx])
            
    if not vectors:
        return None
        
    # Recency-weighted mean: more recent clicks get higher weights.
    # Linear decay: weights = [1, 2, ..., N] / sum
    N = len(vectors)
    weights = np.linspace(0.1, 1.0, N)
    
    # Weighted average
    pooled = np.average(vectors, axis=0, weights=weights)
    
    # L2 normalize
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm
        
    return pooled
