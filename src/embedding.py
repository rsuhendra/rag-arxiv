from __future__ import annotations

from functools import lru_cache
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)

def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return [embedding.astype(float).tolist() for embedding in embeddings]

def embed_query(query: str) -> list[float]:
    model = get_embedding_model()
    embedding = model.encode(query, normalize_embeddings=True, show_progress_bar=False)
    return embedding.astype(float).tolist()
