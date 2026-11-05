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

from pathlib import Path

from app.models import Document
from app.vectordb.chunking import Chunker
from app.vectordb.embeddings import Embedder
from app.vectordb.store import VectorStore

SUPPORTED_SUFFIXES = (".md", ".txt")


def load_document(path: Path) -> Document:
    """IN: path to a .md or .txt file.  OUT: Document(id, text, source=file name, metadata).

    TODO(milestone 6): read the file as UTF-8, build a stable id (e.g. a hash of the
    file name), raise ValueError for unsupported suffixes.
    """
    raise NotImplementedError("TODO(milestone 6): implement load_document in app/ingest/pipeline.py")


def ingest_file(path: Path, *, chunker: Chunker, embedder: Embedder, store: VectorStore) -> int:
    """IN: one file plus the components to use.  OUT: number of chunks stored.

    TODO(milestone 6): load -> chunk -> embed_documents -> delete old copy -> add.
    """
    raise NotImplementedError("TODO(milestone 6): implement ingest_file in app/ingest/pipeline.py")


def ingest_directory(directory: Path, *, chunker: Chunker, embedder: Embedder, store: VectorStore) -> int:
    """IN: a folder.  OUT: total chunks stored across all supported files (sorted by name).

    TODO(milestone 6): loop over supported files and call ingest_file.
    """
    raise NotImplementedError("TODO(milestone 6): implement ingest_directory in app/ingest/pipeline.py")
