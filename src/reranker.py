"""
Cross-encoder reranking — the precision pass after hybrid retrieval.

Why a second model after retrieval already ran: vector/BM25 search score
each chunk against the query *independently* (chunk embedding vs query
embedding, computed separately, then compared). A cross-encoder instead
feeds the (query, chunk) pair into the model *together*, so it can
directly attend to how the two relate. This is much more accurate, but
too slow to run over an entire corpus — which is exactly why we use
cheap retrieval first to narrow thousands of chunks down to ~20-40
candidates, and only run the expensive cross-encoder over that small set.

This two-stage pattern (cheap+broad retrieval, then expensive+narrow
reranking) is standard in production search/RAG systems and is worth
being able to explain clearly — it's a common interview question.
"""

from functools import lru_cache
from sentence_transformers import CrossEncoder

from src.chunker import Chunk
import config


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(config.RERANKER_MODEL_NAME)


def rerank(query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
    """
    Re-scores `chunks` against `query` with a cross-encoder and returns
    the top_k, as [(chunk, rerank_score), ...] sorted descending.
    """
    if not chunks:
        return []

    model = get_reranker()
    pairs = [(query, c.text) for c in chunks]
    scores = model.predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]
