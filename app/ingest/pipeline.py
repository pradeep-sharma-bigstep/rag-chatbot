"""MILESTONE 6 — Ingestion: file -> Document -> Chunks -> vectors -> store.

Used in two places:
  - `uv run python -m app.ingest` indexes everything in data/documents/.
  - The upload endpoint (app/api/documents.py) indexes one uploaded file.

Rules the evaluator checks (with its own chunker/embedder/store):
  - `load_document` supports .md and .txt; anything else raises ValueError.
  - `Document.id` is stable for the same file name (re-loading gives the same id).
  - `ingest_file` returns the number of chunks stored, and the store grows by that much.
  - Ingesting the same file twice does NOT create duplicates
    (hint: `store.delete_document(doc.id)` before adding).
  - `ingest_directory` ingests every supported file and returns the total chunk count.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from app.models import Document
from app.vectordb.chunking import Chunker
from app.vectordb.embeddings import Embedder
from app.vectordb.store import VectorStore

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".md", ".txt")


def _document_id(file_name: str) -> str:
    """Same file name -> same id, so re-ingesting (or re-uploading) a file replaces its old chunks."""
    return "doc-" + hashlib.sha256(file_name.encode("utf-8")).hexdigest()[:16]


def _is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SUFFIXES


def load_document(path: Path) -> Document:
    """IN: path to a .md or .txt file.  OUT: Document(id, text, source=file name, metadata).

    The id depends only on the file *name* (not the folder or the content). Text is read as UTF-8
    (a leading BOM is dropped, Windows line endings become "\\n"). Raises ValueError for an
    unsupported suffix, a path that is not a file, or text that is not valid UTF-8; a missing file
    raises FileNotFoundError.
    """
    path = Path(path)
    if not _is_supported(path):
        raise ValueError(
            f"Unsupported file type {path.suffix!r} for {path.name!r}; expected one of {SUPPORTED_SUFFIXES}"
        )
    if path.exists() and not path.is_file():
        raise ValueError(f"{path} is not a file")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name!r} is not valid UTF-8 text: {exc.reason} at byte {exc.start}") from exc
    return Document(id=_document_id(path.name), text=text, source=path.name, metadata={"suffix": path.suffix.lower()})


def ingest_file(path: Path, *, chunker: Chunker, embedder: Embedder, store: VectorStore) -> int:
    """IN: one file plus the components to use.  OUT: number of chunks stored.

    Safe to run again on the same file: the old chunks are removed first, so the store never holds
    duplicates or leftovers from an earlier, longer version. The embedding step runs *before* anything
    is deleted, so if the embedder fails the previous version of the document stays searchable.
    """
    document = load_document(path)
    chunks = chunker.chunk(document)
    vectors = embedder.embed_documents([chunk.text for chunk in chunks]) if chunks else []
    if len(vectors) != len(chunks):
        raise ValueError(f"Embedder returned {len(vectors)} vectors for {len(chunks)} chunks of {document.source!r}")
    store.delete_document(document.id)  # also clears the old chunks when the file is now empty
    if chunks:
        store.add(chunks, vectors)
    logger.info("Indexed %s: %d chunks", document.source, len(chunks))
    return len(chunks)


def ingest_directory(directory: Path, *, chunker: Chunker, embedder: Embedder, store: VectorStore) -> int:
    """IN: a folder.  OUT: total chunks stored across all supported files (sorted by name).

    Only the folder itself is scanned, not its sub-folders. A file that cannot be ingested (for example
    invalid UTF-8) is logged and skipped, so one bad file does not stop the rest from being indexed.
    Raises NotADirectoryError if `directory` is not a folder.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    total = 0
    for path in sorted(p for p in directory.iterdir() if p.is_file() and _is_supported(p)):
        try:
            total += ingest_file(path, chunker=chunker, embedder=embedder, store=store)
        except ValueError as exc:
            logger.warning("Skipped %s: %s", path.name, exc)
    return total
