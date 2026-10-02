"""
Quick manual test of the retrieval stage, before generation is wired up.

Usage:
    python scripts/test_retrieval.py "does Nimbus support SSO?"
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.keyword_index import load_keyword_index
from src.retriever import hybrid_retrieve


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/test_retrieval.py "your question here"')
        sys.exit(1)

    query = sys.argv[1]
    keyword_index = load_keyword_index()
    results = hybrid_retrieve(query, keyword_index)

    print(f"\nQuery: {query}\n{'-' * 60}")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.chunk.chunk_id}  (rerank_score={r.score:.3f}, rrf_score={r.rrf_score:.4f})")
        preview = r.chunk.text[:350].replace("\n", " ")
        print(f"    {preview}...")


if __name__ == "__main__":
    main()
