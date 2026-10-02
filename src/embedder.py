"""
Wraps the embedding model so the rest of the pipeline never touches
sentence-transformers directly.

Why wrap it: if you later swap embedding models (e.g. to test whether
bge-small beats MiniLM on your eval set), you change one file, not
every place that calls .encode().
"""

from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer

import config


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    # Cached so the (relatively slow) model load only happens once per
    # process, even if get_embedding_model() is called from multiple
    # places (ingestion, query time, eval script).
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts. Returns shape (len(texts), embedding_dim)."""
    model = get_embedding_model()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    return embed_texts([query])[0]
