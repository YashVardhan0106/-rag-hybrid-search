"""
Entry point: run this once (or whenever data/raw/ changes) to (re)build
the vector index.

Usage:
    python scripts/ingest.py
"""

import sys
from pathlib import Path

# Allow running this script directly (python scripts/ingest.py) by
# putting the project root on sys.path, so `import config` / `import src.*`
# resolve regardless of the current working directory.
sys.path.append(str(Path(__file__).parent.parent))

import config
from src.loader import load_directory
from src.chunker import chunk_documents
from src.vector_store import index_chunks
from src.keyword_index import build_keyword_index, save_keyword_index


def main():
    print(f"[ingest] loading documents from {config.DATA_RAW_DIR}")
    documents = load_directory(config.DATA_RAW_DIR)
    print(f"[ingest] loaded {len(documents)} documents")

    chunks = chunk_documents(documents)
    print(f"[ingest] split into {len(chunks)} chunks")

    index_chunks(chunks)
    print(f"[ingest] indexed {len(chunks)} chunks into vector store '{config.CHROMA_COLLECTION_NAME}'")

    keyword_index = build_keyword_index(chunks)
    save_keyword_index(keyword_index)
    print(f"[ingest] built and saved BM25 keyword index ({len(chunks)} chunks)")

    print("[ingest] done.")


if __name__ == "__main__":
    main()
