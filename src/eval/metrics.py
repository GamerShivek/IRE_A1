import numpy as np
import polars as pl
from typing import Callable

def mrr(retrieved: list[str], clicked: set[str]) -> float:
    for i, item in enumerate(retrieved):
        if item in clicked:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(retrieved: list[str], clicked: set[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    dcg = 0.0
    for i, item in enumerate(retrieved_k):
        if item in clicked:
            dcg += 1.0 / np.log2(i + 2)
            
    # idcg
    idcg = 0.0
    num_relevant = min(len(clicked), k)
    for i in range(num_relevant):
        idcg += 1.0 / np.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def auc_from_ranks(retrieved: list[str], clicked: set[str], total_items: int) -> float:
    """
    Compute AUC given a ranked list (top K).
    All un-retrieved items are assumed to be tied at rank K+1.
    """
    k = len(retrieved)
    num_pos = len(clicked)
    num_neg = total_items - num_pos
    
    if num_pos == 0 or num_neg == 0:
        return 0.0
        
    pos_ranks = []
    for i, item in enumerate(retrieved):
        if item in clicked:
            pos_ranks.append(i + 1)
            
    # Unretrieved positives
    num_unretrieved_pos = num_pos - len(pos_ranks)
    for _ in range(num_unretrieved_pos):
        pos_ranks.append(k + 1)
        
    # Sum of ranks of positive items
    sum_pos_ranks = sum(pos_ranks)
    
    # AUC formula based on Mann-Whitney U test
    # U = R1 - n1(n1+1)/2
    # AUC = 1 - U / (n1 * n2)  (since lower rank is better)
    u_stat = sum_pos_ranks - num_pos * (num_pos + 1) / 2.0
    auc = 1.0 - (u_stat / (num_pos * num_neg))
    
    # Clip to [0, 1] just in case
    return max(0.0, min(1.0, auc))

def coverage(all_retrieved_lists: list[list[str]], catalog_size: int) -> float:
    unique_items = set()
    for retrieved in all_retrieved_lists:
        unique_items.update(retrieved)
    return len(unique_items) / catalog_size if catalog_size > 0 else 0.0

def novelty(all_retrieved_lists: list[list[str]], pop_dict: dict[str, int], total_interactions: int) -> float:
    """
    Average negative log probability of recommended items.
    pop_dict: mapping from article_id to its global click count/impressions count.
    total_interactions: sum of all interactions.
    """
    nov_sum = 0.0
    count = 0
    for retrieved in all_retrieved_lists:
        for item in retrieved:
            pop = pop_dict.get(item, 0)
            if pop > 0:
                prob = pop / total_interactions
                nov_sum += -np.log2(prob)
            else:
                # Max novelty for unseen items
                nov_sum += -np.log2(1.0 / total_interactions) if total_interactions > 0 else 0.0
            count += 1
    return nov_sum / count if count > 0 else 0.0

def intra_list_diversity(retrieved: list[str], embeddings_dict: dict[str, np.ndarray]) -> float:
    """
    Average pairwise distance between items in the list.
    distance = 1 - cosine_similarity.
    """
    if len(retrieved) < 2:
        return 0.0
        
    vecs = []
    for item in retrieved:
        if item in embeddings_dict:
            vecs.append(embeddings_dict[item])
            
    if len(vecs) < 2:
        return 0.0
        
    matrix = np.vstack(vecs)
    # Cosine similarity (assuming vectors are L2 normalized)
    sim_matrix = matrix @ matrix.T
    
    # Distance
    dist_matrix = 1.0 - sim_matrix
    
    # Average upper triangle
    idx = np.triu_indices(len(vecs), k=1)
    avg_dist = float(np.mean(dist_matrix[idx]))
    return avg_dist

def bootstrap_ci(metric_fn: Callable, data: list, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float, float]:
    """
    Returns (mean, lower_bound, upper_bound)
    """
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0
        
    values = []
    for _ in range(n_bootstrap):
        # sample with replacement
        indices = np.random.randint(0, n, size=n)
        sample = [data[i] for i in indices]
        values.append(metric_fn(sample))
        
    mean = np.mean(values)
    lower = np.percentile(values, (1.0 - ci) / 2 * 100)
    upper = np.percentile(values, (1.0 + ci) / 2 * 100)
    
    return float(mean), float(lower), float(upper)
