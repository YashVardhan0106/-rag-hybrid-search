"""
Loads raw documents from disk into a common in-memory representation.

Why a separate loader module: chunking/embedding shouldn't care whether
a document came from a .md file, a PDF, or a database export. This
module's job is to normalize everything into a simple `Document` shape
before it hits the rest of the pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from pypdf import PdfReader


@dataclass
class Document:
    """A single loaded document, before chunking."""
    text: str
    source: str                      # filename, used for citations later
    metadata: dict = field(default_factory=dict)


def load_markdown(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    return Document(text=text, source=path.name, metadata={"type": "markdown"})


def load_pdf(path: Path) -> Document:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages)
    return Document(
        text=text,
        source=path.name,
        metadata={"type": "pdf", "num_pages": len(reader.pages)},
    )


def load_directory(dir_path: Path) -> list[Document]:
    """
    Load every supported file in a directory.

    Returns a flat list of Documents. Unsupported file types are skipped
    with a printed warning rather than crashing the whole ingestion run —
    a real document corpus will always have a few stray files in it.
    """
    documents = []
    for path in sorted(dir_path.iterdir()):
        if path.suffix.lower() in (".md", ".markdown", ".txt"):
            documents.append(load_markdown(path))
        elif path.suffix.lower() == ".pdf":
            documents.append(load_pdf(path))
        else:
            print(f"[loader] skipping unsupported file: {path.name}")
    return documents
