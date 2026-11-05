"""MILESTONE 4 — Embeddings.

Turn text into vectors so similar meaning -> nearby vectors.

Pick any embedding model: a hosted API (OpenAI, Voyage, Cohere ...) or a local
one (sentence-transformers, fastembed ...). Add its package with `uv add <pkg>`
and read the model name from `EMBEDDING_MODEL` in `.env`.

Rules the evaluator checks:
  - `embed_documents(texts)` returns one vector per text, in the same order.
  - Every vector has length `dimension`.
  - `embed_documents([])` returns `[]`.
  - The same text always gives (almost) the same vector.
  - `embed_query(t)` is close to `embed_documents([t])[0]`.
  - Related sentences are closer (cosine) than unrelated ones.
"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    @property
    def dimension(self) -> int:
        """Length of every vector this embedder returns."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """IN: list of texts.  OUT: list of vectors (same length and order)."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """IN: one search query.  OUT: one vector."""
        ...


def get_embedder() -> Embedder:
    """Return the configured embedder.

    TODO(milestone 4):
      - Write a class that satisfies `Embedder` (wrap your chosen model/API).
      - Read the model name from `app.config.get_settings().embedding_model`.
      - Batch calls to the model; don't call it once per text.
    """
    raise NotImplementedError("TODO(milestone 4): implement get_embedder in app/vectordb/embeddings.py")
