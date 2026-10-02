"""
End-to-end RAG query: retrieve, then generate.

This is the function the Streamlit UI (Step 5) will call directly —
scripts/query.py exists both as a CLI tool for manual testing and as
the reference implementation of "how do the pieces get chained together."

Usage:
    python scripts/query.py "does Nimbus support SSO?"
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.keyword_index import load_keyword_index
from src.retriever import hybrid_retrieve
from src.generator import generate_answer


def ask(query: str, keyword_index=None):
    """
    The single function the UI layer will call. Loads the keyword index
    lazily if not passed in (so callers that query repeatedly, like the
    Streamlit app, can load it once and reuse it instead of re-reading
    the pickle file on every question).
    """
    if keyword_index is None:
        keyword_index = load_keyword_index()

    retrieved = hybrid_retrieve(query, keyword_index)
    answer = generate_answer(query, retrieved)
    return answer, retrieved


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/query.py "your question here"')
        sys.exit(1)

    query = sys.argv[1]
    answer, retrieved = ask(query)

    print(f"\nQuestion: {query}\n{'-' * 60}")
    print(f"\n{answer.text}\n")
    print(f"Sources: {', '.join(answer.sources)}")

    print(f"\n{'-' * 60}\nRetrieved chunks (for debugging):")
    for i, r in enumerate(retrieved, 1):
        print(f"  [{i}] {r.chunk.chunk_id}  (score={r.score:.3f})")


if __name__ == "__main__":
    main()
