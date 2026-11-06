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

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from app.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions, ~70 MB, runs on the CPU, no API key
BATCH_SIZE = 32  # texts sent to the model per call


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


def _cache_dir() -> str:
    """Where model files are kept. fastembed's own default is a temp folder that the OS clears."""
    return os.getenv("FASTEMBED_CACHE_PATH") or str(Path.home() / ".cache" / "fastembed")


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> Any:
    """Load (and on first use download) a fastembed model. Cached: loading takes a moment."""
    from fastembed import TextEmbedding

    logger.info("Loading embedding model %s", model_name)
    try:
        return TextEmbedding(model_name=model_name, cache_dir=_cache_dir())
    except ValueError as exc:  # fastembed does not know this model name
        raise ValueError(f"Unsupported EMBEDDING_MODEL {model_name!r}: {exc}") from exc


class FastEmbedEmbedder:
    """Local embeddings with fastembed (ONNX models on the CPU, no API key, works offline once downloaded).

    Documents and queries go through the model's own `passage_embed` / `query_embed`, so models that
    expect a different prefix for questions (e5, some bge variants) are handled by the library.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, model: Any | None = None) -> None:
        self.model_name = model_name
        self._model = model if model is not None else _load_model(model_name)
        self._dimension = int(self._model.embedding_size)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.passage_embed(texts, batch_size=BATCH_SIZE)  # one vector per text, in order
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = next(iter(self._model.query_embed(text)))
        vector_list: list[float] = vector.tolist()
        return vector_list


def get_embedder() -> Embedder:
    """Return the configured embedder.

    The model name comes from EMBEDDING_MODEL in `.env`; when it is empty, `DEFAULT_MODEL` is used.
    The first call downloads the model (a few seconds, needs internet once); later calls reuse it.
    Raises ValueError for a model name fastembed does not support.
    """
    model_name = get_settings().embedding_model.strip() or DEFAULT_MODEL
    return FastEmbedEmbedder(model_name)
