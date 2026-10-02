"""
Ties retrieval together: vector search + BM25 keyword search, merged,
then reranked down to a final top-k.

The merge step deserves explanation. Vector search returns cosine
similarities (0 to 1); BM25 returns unbounded term-frequency scores
(could be 0, could be 15). You cannot directly compare or add these two
scores together — a BM25 score of 8 doesn't mean "8x more relevant"
than a vector score of 0.8, they're not on the same scale at all.

Two common fixes:
  1. Normalize both score sets (e.g. min-max) before combining.
  2. Skip score comparison entirely and merge by *rank* instead of
     *score*, using something like Reciprocal Rank Fusion (RRF).

This implementation uses RRF because it's simpler, has no tunable
weights to get wrong, and is what most production hybrid-search systems
(Elasticsearch, Weaviate, Azure AI Search) use under the hood.
"""

from dataclasses import dataclass

from src.chunker import Chunk
from src.embedder import embed_query
from src.vector_store import query_vector_store
from src.keyword_index import KeywordIndex, query_keyword_index
from src.reranker import rerank
import config


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float               # final reranked score
    rrf_score: float           # score after hybrid fusion, before reranking


def _reciprocal_rank_fusion(
    vector_ranked: list[str],      # list of chunk_ids, best first
    bm25_ranked: list[str],        # list of chunk_ids, best first
    k: int = 60,                   # RRF damping constant (60 is the standard default)
) -> dict[str, float]:
    """
    Combines two ranked lists into one fused score per chunk_id, using
    Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across the lists
    a chunk_id appears in. Being ranked highly in *either* list, or
    moderately in *both*, pushes a chunk's fused score up.
    """
    fused_scores: dict[str, float] = {}
    for ranked_list in (vector_ranked, bm25_ranked):
        for rank, chunk_id in enumerate(ranked_list):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused_scores


def hybrid_retrieve(query: str, keyword_index: KeywordIndex) -> list[RetrievedChunk]:
    # --- Stage 1: cheap, broad retrieval from two independent sources ---
    query_embedding = embed_query(query)
    vector_results = query_vector_store(query_embedding, top_k=config.VECTOR_TOP_K)
    vector_chunk_ids = vector_results["ids"][0]           # best-first, already ranked

    bm25_results = query_keyword_index(keyword_index, query, top_k=config.BM25_TOP_K)
    bm25_chunk_ids = [c.chunk_id for c, _score in bm25_results]

    # --- Stage 2: fuse the two rankings ---
    fused_scores = _reciprocal_rank_fusion(vector_chunk_ids, bm25_chunk_ids)

    # Build a lookup so we can go from chunk_id back to a full Chunk object.
    # bm25_results already carries Chunk objects; vector results only carry
    # ids/text/metadata separately, so we prefer the bm25 lookup when
    # available and fall back to reconstructing from vector_results.
    chunk_lookup = {c.chunk_id: c for c, _score in bm25_results}
    for i, chunk_id in enumerate(vector_chunk_ids):
        if chunk_id not in chunk_lookup:
            chunk_lookup[chunk_id] = Chunk(
                text=vector_results["documents"][0][i],
                source=vector_results["metadatas"][0][i].get("source", "unknown"),
                chunk_id=chunk_id,
                metadata=vector_results["metadatas"][0][i],
            )

    fused_ranked = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
    candidate_chunks = [chunk_lookup[chunk_id] for chunk_id, _score in fused_ranked]

    # --- Stage 3: expensive, narrow reranking on the fused candidate set ---
    reranked = rerank(query, candidate_chunks, top_k=config.RERANK_TOP_K)

    rrf_lookup = dict(fused_ranked)
    return [
        RetrievedChunk(chunk=chunk, score=float(score), rrf_score=rrf_lookup[chunk.chunk_id])
        for chunk, score in reranked
    ]
