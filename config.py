"""
Central configuration for the RAG pipeline.

Keeping every tunable value in one file (instead of scattered magic
numbers across scripts) means:
  - you can experiment by changing one place
  - your eval script can log exactly which config produced which score
  - it reads well in a code review / interview walkthrough
"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
CHROMA_COLLECTION_NAME = "documents"

# --- Chunking ---
CHUNK_SIZE = 120          # tokens (approx, via character heuristic)
CHUNK_OVERLAP = 50        # tokens of overlap between adjacent chunks

# --- Embedding model ---
# all-MiniLM-L6-v2: 384-dim, ~80MB, fast on CPU, strong baseline quality.
# Swap to "BAAI/bge-small-en-v1.5" for a quality bump at similar speed.
# This runs 100% locally — no API cost, no rate limits during ingestion —
# which is why it stays local even though generation below uses a hosted API.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Retrieval ---
VECTOR_TOP_K = 20         # how many candidates vector search returns
BM25_TOP_K = 20           # how many candidates keyword search returns
RERANK_TOP_K = 5          # how many chunks survive reranking, go to the LLM

# --- Reranking ---
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Generation ---
# Google AI Studio's free tier (Gemini API) is the only major provider with
# a genuinely ongoing, no-credit-card-required free tier — good fit for a
# self-funded portfolio project. Flash models are the free-tier workhorse.
#
# Google ships new Flash point-releases every few weeks, and older ones get
# shut down on a schedule (check https://ai.google.dev/gemini-api/docs/models
# before relying on this in the future). Update the string below if this
# model has been deprecated by the time you're reading this.
LLM_MODEL = "gemini-flash-latest"
LLM_MAX_TOKENS = 1024
