"""MILESTONE 7 — Retriever.

Glue between a question and the vector store: embed the query, search, and
(optionally) post-process — score threshold, de-duplication, MMR, re-ranking.

Rules the evaluator checks:
  - `retrieve(query)` returns at most `k` RetrievedChunk, best first.
  - `retrieve(query, k=2)` overrides the default k.
  - It uses the embedder and store it was GIVEN (the evaluator passes its own).
  - It shows up in LangSmith as a run with run_type="retriever"
    (hint: `from langsmith import traceable` and `@traceable(run_type="retriever")`).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

from langsmith import traceable

from app.config import get_settings
from app.models import RetrievedChunk
from app.vectordb.embeddings import Embedder, get_embedder
from app.vectordb.store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """IN: a user question, optional k.  OUT: the most relevant chunks."""
        ...


def _as_documents(hits: list[RetrievedChunk]) -> dict[str, Any]:
    """Shape the hits the way LangSmith expects a retriever's output, so it shows them as documents."""
    documents = [
        {
            "page_content": hit.chunk.text,
            "type": "Document",
            "metadata": {"source": hit.chunk.source, "chunk_id": hit.chunk.id, "score": hit.score},
        }
        for hit in hits
    ]
    return {"documents": documents}


class StoreRetriever:
    """Embed the question, search the vector store, return the best chunks.

    `min_score` (optional) drops weak matches. It is off by default because scores depend on the
    embedding model: with the default bge-small model, relevant chunks score about 0.55-0.8 and
    unrelated small talk ("hello") about 0.45-0.55, so a cut-off near 0.6 is a reasonable start.
    """

    def __init__(self, store: VectorStore, embedder: Embedder, k: int, min_score: float | None = None) -> None:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.store = store
        self.embedder = embedder
        self.k = k
        self.min_score = min_score

    @traceable(run_type="retriever", name="retrieve", process_outputs=_as_documents)
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        limit = self.k if k is None else k
        if limit < 1 or not query.strip() or self.store.count() == 0:
            return []  # nothing to look for, or nothing to look in (skips the embedding work)
        hits = self.store.search(self.embedder.embed_query(query), limit)
        if self.min_score is not None:
            hits = [hit for hit in hits if hit.score >= self.min_score]
        return hits


def _min_score_from_env() -> float | None:
    raw = os.getenv("RETRIEVAL_MIN_SCORE", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring RETRIEVAL_MIN_SCORE=%r: not a number", raw)
        return None


def get_retriever(
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    k: int | None = None,
) -> Retriever:
    """Build a retriever.

    IN:  store / embedder to use (None -> build the defaults with
         get_vector_store() / get_embedder()), default k (None -> settings.top_k).
    OUT: an object with `.retrieve(query, k=None)`.

    Optional: set RETRIEVAL_MIN_SCORE in `.env` to drop matches scoring below it.
    """
    return StoreRetriever(
        store=store if store is not None else get_vector_store(),
        embedder=embedder if embedder is not None else get_embedder(),
        k=k if k is not None else get_settings().top_k,
        min_score=_min_score_from_env(),
    )
