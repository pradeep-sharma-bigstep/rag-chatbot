"""MILESTONE 5 — Vector store.

Store chunk vectors on disk and find the nearest ones to a query vector.

Use any backend: Chroma, FAISS, LanceDB, Qdrant (local mode) ... or your own
NumPy + JSON file. Add the package with `uv add <pkg>`. Data must live under
`persist_dir` (default: settings.vector_dir) — never a hard-coded path.

Rules the evaluator checks:
  - `search(v, k)` returns at most k hits, best first (scores descending).
  - Searching with a stored chunk's own vector returns that chunk first.
  - `add` with an existing chunk id REPLACES it (no duplicates).
  - A new store opened on the same `persist_dir` sees the same data.
  - `delete_document(doc_id)` removes all chunks of that document.
  - Searching an empty store returns [].
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.models import Chunk, RetrievedChunk


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """IN: chunks and their vectors (same length and order). Upsert by chunk.id."""
        ...

    def search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        """IN: a query vector and k.  OUT: up to k RetrievedChunk, highest score first."""
        ...

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk whose doc_id matches. OUT: how many were removed."""
        ...

    def count(self) -> int:
        """How many chunks are stored."""
        ...


def get_vector_store(persist_dir: str | Path | None = None) -> VectorStore:
    """Open (or create) the vector store saved in `persist_dir`.

    TODO(milestone 5):
      - Write a class that satisfies `VectorStore` using your chosen backend.
      - Default `persist_dir` to `app.config.get_settings().vector_dir`.
      - Keep the Chunk fields (text, source, position, metadata) so `search`
        can rebuild full Chunk objects.
    """
    raise NotImplementedError("TODO(milestone 5): implement get_vector_store in app/vectordb/store.py")
