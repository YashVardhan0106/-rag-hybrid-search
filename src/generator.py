"""
Calls the Gemini API to generate a grounded answer from retrieved chunks.

Loading the API key: we use python-dotenv here so GEMINI_API_KEY can
live in a local .env file (gitignored) instead of being hardcoded or
set manually in every terminal session. The google-genai client also
auto-detects GEMINI_API_KEY once it's in the environment — we don't
pass it explicitly, we just need it to already be loaded by the time
Client() is constructed.

Retry logic: free-tier Gemini traffic gets lower priority than paid
traffic, so 503 UNAVAILABLE ("model is overloaded") and 429 RESOURCE_EXHAUSTED
("rate limited") are expected, transient conditions — not bugs. Retrying
with exponential backoff (wait longer after each failure) is the standard
way to ride these out instead of surfacing a crash to the user on the
first hiccup. We deliberately do NOT retry on other error types (e.g. a
404 for a bad model name, or a 400 for a malformed request) — those are
real bugs that retrying won't fix, so we let them raise immediately.
"""

import time
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from src.retriever import RetrievedChunk
from src.prompt import build_prompt
import config

load_dotenv()  # reads .env into the process environment, once, at import time

# Error codes worth retrying: both are transient server/capacity conditions.
_RETRYABLE_CODES = {503, 429}
_MAX_RETRIES = 4
_BASE_DELAY_SECONDS = 2  # doubles each retry: 2s, 4s, 8s, 16s

# We never pass tools/functions, so disable AFC to avoid the SDK's
# "Direct use of AFC..." warning on every single call.
_GENERATION_CONFIG = genai_types.GenerateContentConfig(
    max_output_tokens=config.LLM_MAX_TOKENS,
    automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
)

@dataclass
class GeneratedAnswer:
    text: str
    sources: list[str]          # deduplicated source filenames actually shown to the model


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    # Cached for the same reason as the embedding/reranker models: avoid
    # reconstructing the client on every call within a process.
    return genai.Client()


def _call_with_retry(client: genai.Client, prompt: str):
    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            return client.models.generate_content(
                model=config.LLM_MODEL,
                contents=prompt,
                config=_GENERATION_CONFIG,
            )
        except genai_errors.ServerError as e:
            code = getattr(e, "code", None)
            if code not in _RETRYABLE_CODES:
                raise  # not a transient error — don't retry, surface it immediately

            last_error = e
            if attempt < _MAX_RETRIES - 1:
                delay = _BASE_DELAY_SECONDS * (2 ** attempt)
                print(f"[generator] Gemini returned {code} (attempt {attempt + 1}/{_MAX_RETRIES}); "
                      f"retrying in {delay}s...")
                time.sleep(delay)

    # Exhausted all retries — raise the last error we saw rather than
    # swallowing it, so the caller (and the user) knows generation failed.
    raise last_error


def generate_answer(query: str, retrieved: list[RetrievedChunk]) -> GeneratedAnswer:
    if not retrieved:
        return GeneratedAnswer(
            text="I couldn't find anything relevant in the knowledge base to answer that.",
            sources=[],
        )

    prompt = build_prompt(query, retrieved)
    client = get_client()

    response = _call_with_retry(client, prompt)

    sources = list(dict.fromkeys(r.chunk.source for r in retrieved))  # dedupe, preserve order
    return GeneratedAnswer(text=response.text, sources=sources)
