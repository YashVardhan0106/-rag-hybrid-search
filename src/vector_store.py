"""
Thin wrapper around ChromaDB.

Chroma is used here as an embedded, persistent vector store — no
separate server process to run, which keeps this demo-able with a
single `streamlit run`. In production you'd likely swap this for
pgvector or a managed vector DB (Pinecone/Weaviate), but the interface
below (add / query) would stay the same, so only this file would change.
"""

import chromadb
from chromadb.config import Settings

from src.chunker import Chunk
from src.embedder import embed_texts
import config


def get_collection():
    client = chromadb.PersistentClient(
        path=str(config.CHROMA_DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[Chunk]) -> None:
    """Embed a list of chunks and upsert them into the vector store."""
    collection = get_collection()
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{"source": c.source, **c.metadata} for c in chunks],
    )


def query_vector_store(query_embedding, top_k: int) -> dict:
    """Return the top_k most similar chunks to a query embedding."""
    collection = get_collection()
    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )
