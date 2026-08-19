import logging
import numpy as np

log = logging.getLogger(__name__)

class BruteForceIndex:
    def __init__(self, article_ids: list[str], vectors: np.ndarray):
        """
        vectors should be L2 normalized.
        """
        self.article_ids = article_ids
        self.vectors = vectors
        self.id_to_idx = {aid: i for i, aid in enumerate(article_ids)}

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        query_vectors: (N, D) array of L2 normalized user vectors.
        returns: (distances, indices)
                 indices is (N, k) integer array of indices into self.article_ids.
        """
        # Exact cosine similarity = dot product if vectors are L2 normalized.
        # scores: (N, num_articles)
        scores = query_vectors @ self.vectors.T
        
        # We want top-k highest scores, which means sorting descending.
        # np.argpartition is faster than full sort.
        if k >= self.vectors.shape[0]:
            k = self.vectors.shape[0]
            
        # To sort descending, we negate scores. Or use -scores.
        indices = np.argpartition(-scores, kth=k-1, axis=1)[:, :k]
        
        # Sort the top-k exactly
        # We need to sort indices according to the actual scores
        for i in range(len(indices)):
            row_idx = indices[i]
            row_scores = scores[i, row_idx]
            # sort ascending, then reverse
            sorted_k = np.argsort(row_scores)[::-1]
            indices[i] = row_idx[sorted_k]
            
        # extract distances (similarities)
        distances = np.take_along_axis(scores, indices, axis=1)
        
        return distances, indices

def build_index(article_ids: list[str], vectors: np.ndarray) -> BruteForceIndex:
    log.info(f"Building exact brute-force index with {len(article_ids)} articles.")
    return BruteForceIndex(article_ids, vectors)
