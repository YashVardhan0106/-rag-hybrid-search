# Hybrid RAG Document Q&A System

A retrieval-augmented generation (RAG) pipeline that answers questions over a
document corpus using **hybrid search** (dense vector + BM25 keyword),
**cross-encoder reranking**, and **citation-grounded generation** — built to
demonstrate production-oriented RAG architecture, not just a tutorial-level
"embed and ask" demo.

## Why this exists

Most RAG tutorials stop at "embed chunks, do a vector search, stuff into a
prompt." That approach has real, well-known gaps: pure vector search misses
exact keyword matches (product names, acronyms, IDs), and similarity-ranked
results don't reliably put the *most* relevant chunk first. This project
addresses both directly, with an architecture that mirrors patterns used in
production search/RAG systems rather than the simplest thing that works on a
toy example.

## Architecture

```
INGESTION (run once)
  documents → chunk → ┬→ embed → ChromaDB (vector index)
                       └→ tokenize → BM25 index (pickled)

QUERY (per question)
  question → embed ─────────────┐
           → tokenize ──────────┤
                                 ▼
                    vector search (top 20) ─┐
                    BM25 search (top 20) ───┤
                                 ▼
                   Reciprocal Rank Fusion (merge by rank,
                   not raw score — the two scales aren't comparable)
                                 ▼
                   cross-encoder reranking (top 3)
                                 ▼
                   citation-tagged prompt assembly
                                 ▼
                   Gemini API generation (retry + backoff
                   on transient failures)
                                 ▼
                   answer + cited sources
```

## Key design decisions

- **Hybrid retrieval over pure vector search.** Dense embeddings are strong
  on semantic/paraphrase matches but weak on exact tokens with little
  semantic content (acronyms, IDs, specific numbers). BM25 covers that gap.
- **Reciprocal Rank Fusion, not score averaging.** Cosine similarity (0–1)
  and BM25 scores (unbounded, corpus-dependent) aren't on comparable scales.
  RRF merges by rank position instead, avoiding a normalization step with no
  principled way to weight the two methods.
- **Cross-encoder reranking as a second stage.** Bi-encoders (used for vector
  search) compare independently-computed embeddings; a cross-encoder reads
  the query and each candidate chunk *together*, producing far more precise
  relevance scores — at a cost too high to run over an entire corpus, which
  is why it only runs on the small, pre-filtered candidate set from hybrid
  retrieval.
- **Explicit anti-hallucination prompting.** The generation prompt instructs
  the model to answer only from retrieved context and say so when the
  context doesn't contain an answer — without this, an LLM silently falls
  back on its own training knowledge and the entire point of grounding is
  lost.
- **Provider-agnostic generation layer.** The LLM backend has been swapped
  across three providers (Anthropic → OpenAI → Gemini) by changing only
  `config.py` and `src/generator.py` — nothing in retrieval, chunking, or
  embedding needed to change.
- **Resilience for transient failures.** API calls are wrapped with
  exponential-backoff retry limited to genuinely transient error codes
  (503/429) — real errors (bad model name, malformed request) still fail
  immediately instead of retrying pointlessly.

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.10+ |
| Document loading | `pypdf`, markdown |
| Chunking | `langchain-text-splitters` (recursive, structure-aware) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, local, free) |
| Vector store | ChromaDB (persistent, embedded) |
| Keyword search | `rank-bm25` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | Gemini API (`google-genai`) |
| UI | Streamlit |

## Project structure

```
rag-project/
├── config.py              # all tunables in one place
├── data/raw/               # source documents
├── src/
│   ├── loader.py            # file → Document
│   ├── chunker.py           # Document → Chunk[]
│   ├── embedder.py          # local embedding model wrapper
│   ├── vector_store.py      # ChromaDB wrapper
│   ├── keyword_index.py     # BM25 index
│   ├── reranker.py          # cross-encoder reranking
│   ├── retriever.py         # hybrid retrieval + RRF orchestration
│   ├── prompt.py             # citation-aware prompt assembly
│   └── generator.py         # Gemini call + retry/backoff
└── scripts/
    ├── ingest.py             # build the indexes
    ├── test_retrieval.py     # inspect retrieval in isolation
    ├── query.py              # end-to-end CLI query
    └── list_models.py        # debug available Gemini models
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env             # add your GEMINI_API_KEY
```

## Usage

```bash
python scripts/ingest.py                              # build indexes (run once)
python scripts/query.py "does the product support SSO?"
```

## What's next (evaluation)

A small hand-built set of (question, expected source) pairs to measure
retrieval precision/recall, plus a lightweight generation-faithfulness check
(does the answer actually reflect retrieved context) — the piece most RAG
tutorials skip entirely, and the one that most demonstrates whether the
system is actually trustworthy rather than merely plausible.
