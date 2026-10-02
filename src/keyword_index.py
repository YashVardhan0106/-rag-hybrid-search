"""
BM25 keyword search — the lexical half of hybrid retrieval.

Why this matters alongside vector search: embeddings are great at
matching *meaning* ("how much does the cheapest plan cost" ~ "Starter
pricing") but can be weak on exact tokens that carry little semantic
weight on their own — product names, error codes, version numbers,
acronyms like "SSO" or "BYOK". A question like "does Nimbus support
SAML 2.0" is best answered by a keyword hit on the literal string
"SAML 2.0", which BM25 is very good at and dense embeddings sometimes
aren't.

Unlike the vector store, BM25 has no persistent index of its own here —
it's cheap to rebuild in memory at query time from the same chunks
that were embedded, so we do that rather than adding a second
persistence layer to keep in sync with Chroma.
"""

import pickle
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.chunker import Chunk
import config


@dataclass
class KeywordIndex:
    bm25: BM25Okapi
    chunks: list[Chunk]           # parallel list — bm25's internal order


def _tokenize(text: str) -> list[str]:
    # Simple, fast tokenizer: lowercase, strip punctuation, split on
    # whitespace. BM25 doesn't need anything fancier than this — it's
    # scoring term overlap, not doing semantic understanding.
    return re.findall(r"[a-z0-9]+", text.lower())


def build_keyword_index(chunks: list[Chunk]) -> KeywordIndex:
    tokenized_corpus = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return KeywordIndex(bm25=bm25, chunks=chunks)


def save_keyword_index(index: KeywordIndex, path=None) -> None:
    path = path or (config.DATA_PROCESSED_DIR / "keyword_index.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f)


def load_keyword_index(path=None) -> KeywordIndex:
    path = path or (config.DATA_PROCESSED_DIR / "keyword_index.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def query_keyword_index(index: KeywordIndex, query: str, top_k: int) -> list[tuple[Chunk, float]]:
    """Returns [(chunk, bm25_score), ...] sorted by score descending."""
    tokenized_query = _tokenize(query)
    scores = index.bm25.get_scores(tokenized_query)
    ranked = sorted(zip(index.chunks, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]
