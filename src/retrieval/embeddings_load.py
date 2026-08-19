import logging
import os
import polars as pl
import numpy as np

log = logging.getLogger(__name__)

def load_or_compute_embeddings(
    dataset: str,
    articles_df: pl.DataFrame,
    ebnerd_emb_type: str = "word2vec",
    model_name: str = "all-MiniLM-L6-v2"
) -> tuple[list[str], np.ndarray]:
    """
    Returns (article_ids, embeddings_matrix).
    Embeddings are L2 normalized.
    """
    if dataset == "ebnerd":
        if ebnerd_emb_type == "mbert":
            emb_path = "data/raw/google_bert_base_multilingual_cased/google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet"
        else:
            emb_path = "data/raw/Ekstra_Bladet_word2vec/Ekstra_Bladet_word2vec/document_vector.parquet"
            
        log.info(f"Loading provided EB-NeRD {ebnerd_emb_type} embeddings from {emb_path}...")
        emb_df = pl.read_parquet(emb_path)
        if ebnerd_emb_type == "mbert":
            emb_df = emb_df.rename({"google-bert/bert-base-multilingual-cased": "document_vector"})
        emb_df = emb_df.with_columns(
            ("ebnerd_" + pl.col("article_id").cast(pl.Utf8)).alias("article_id")
        )
        
        # We must ensure we cover all articles in articles_df
        # Join with articles_df to keep the exact same order and coverage
        catalog_ids = articles_df.select("article_id")
        merged = catalog_ids.join(emb_df, on="article_id", how="left")
        
        # Handle missing vectors by filling with zeros
        null_count = merged.filter(pl.col("document_vector").is_null()).height
        if null_count > 0:
            log.warning(f"Found {null_count} articles without provided embeddings, filling with zeros.")
            # Fill with zeros. We need to know the vector dimension.
            dim = len(emb_df["document_vector"][0])
            zero_vec = [0.0] * dim
            merged = merged.with_columns(
                pl.col("document_vector").fill_null(pl.lit(zero_vec))
            )
        
        vectors = np.vstack(merged["document_vector"].to_list())
        article_ids = merged["article_id"].to_list()
        
    elif dataset == "mind":
        log.info(f"Computing MIND embeddings using sentence-transformers model: {model_name}...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        
        titles = articles_df["title"].fill_null("").to_list()
        abstracts = articles_df["abstract"].fill_null("").to_list()
        texts = [f"{t} {a}".strip() for t, a in zip(titles, abstracts)]
        
        log.info(f"Encoding {len(texts)} articles. This might take a bit...")
        vectors = model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
        article_ids = articles_df["article_id"].to_list()
        
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # L2 normalize all vectors (sentence-transformers does it if normalize_embeddings=True, but we do it anyway to be safe, especially for word2vec)
    log.info("L2 normalizing embeddings...")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero
    vectors = vectors / norms
    
    return article_ids, vectors
