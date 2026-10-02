"""
Builds the prompt sent to the LLM from retrieved chunks.

Kept as its own module (rather than inlined in generator.py) because
prompt wording is something you'll iterate on a lot once you start
testing real questions — having it isolated makes that easy to do
without touching the API-calling code at all, and makes it easy to
show "before/after" prompt versions in a portfolio writeup.

Design choices worth being able to explain:
  - Each chunk is labeled with a citation tag like [1], [2] so the model
    can reference exactly which source backed a claim.
  - The system instruction explicitly tells the model to say when the
    context doesn't answer the question, instead of falling back to its
    own general knowledge — this is the core anti-hallucination guardrail
    in a RAG system. Without it, an LLM will happily answer from its
    training data and you lose the entire point of grounding it in your
    documents.
"""

from src.retriever import RetrievedChunk

SYSTEM_INSTRUCTION = """You are a helpful assistant answering questions using ONLY the provided context.

Rules:
1. Base your answer strictly on the context below. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say so explicitly — do not guess or make something up.
3. When you state a fact, cite the source using its bracketed number, e.g. [1].
4. Keep answers concise and direct."""


def format_context(retrieved: list[RetrievedChunk]) -> str:
    """Turns retrieved chunks into a numbered, citable context block."""
    lines = []
    for i, r in enumerate(retrieved, 1):
        lines.append(f"[{i}] (source: {r.chunk.source})\n{r.chunk.text}")
    return "\n\n".join(lines)


def build_prompt(query: str, retrieved: list[RetrievedChunk]) -> str:
    context = format_context(retrieved)
    return f"""{SYSTEM_INSTRUCTION}

Context:
{context}

Question: {query}

Answer:"""
