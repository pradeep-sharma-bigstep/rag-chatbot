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

from typing import Protocol

from app.models import RetrievedChunk
from app.vectordb.embeddings import Embedder
from app.vectordb.store import VectorStore


class Retriever(Protocol):
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """IN: a user question, optional k.  OUT: the most relevant chunks."""
        ...


def get_retriever(
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    k: int | None = None,
) -> Retriever:
    """Build a retriever.

    IN:  store / embedder to use (None -> build the defaults with
         get_vector_store() / get_embedder()), default k (None -> settings.top_k).
    OUT: an object with `.retrieve(query, k=None)`.

    TODO(milestone 7): implement a Retriever class and return it here.
    """
    raise NotImplementedError("TODO(milestone 7): implement get_retriever in app/vectordb/retriever.py")
