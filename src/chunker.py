"""
Splits loaded documents into retrieval-sized chunks.

Strategy: recursive character splitting. Instead of blindly cutting text
every N characters (which can slice a sentence in half and hand the LLM
a fragment like "customers who exceed their monthly row lim"), we try
to split on paragraph boundaries first, then sentence boundaries, then
words — only falling back to a hard character cut if nothing else works.

This is the single highest-leverage place to invest quality in a RAG
pipeline: retrieval can only ever be as good as the chunks it searches
over. Bad chunking silently caps your ceiling on everything downstream.
"""

from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.loader import Document
import config


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: str                     # e.g. "01_product_overview.md::2"
    metadata: dict = field(default_factory=dict)


def chunk_document(doc: Document) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE * 4,      # ~4 chars per token, rough heuristic
        chunk_overlap=config.CHUNK_OVERLAP * 4,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(doc.text)

    chunks = []
    for i, text in enumerate(raw_chunks):
        chunks.append(
            Chunk(
                text=text.strip(),
                source=doc.source,
                chunk_id=f"{doc.source}::{i}",
                metadata={**doc.metadata, "chunk_index": i},
            )
        )
    return chunks


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
