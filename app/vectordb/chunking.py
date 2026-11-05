"""MILESTONE 3 — Chunking.

Split a Document into Chunks small enough to embed and retrieve.

You choose the METHOD (fixed-size with overlap, recursive by paragraph/sentence,
markdown-heading aware, semantic ...). The method is picked by name from
`CHUNKING_STRATEGY` in `.env`, so you can add several and switch between them.

Whatever method you use, every chunker must obey these rules (the evaluator
checks them on documents you have never seen):

  1. An empty / whitespace-only document gives `[]`.
  2. No chunk is empty or whitespace-only.
  3. `len(chunk.text) <= chunk_size` (characters) for every chunk.
  4. No text is lost: every word of the document appears in at least one chunk.
  5. `chunk.doc_id == document.id` and `chunk.source == document.source`.
  6. `position` is 0, 1, 2 ... in document order; `id` is unique (e.g. f"{doc_id}:{position}").
  7. Same input -> same output (no randomness).
  8. A smaller `chunk_size` gives more chunks for the same long document.
"""

from __future__ import annotations

from typing import Protocol

from app.models import Chunk, Document


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]:
        """IN: one Document.  OUT: its Chunks, in order."""
        ...


def get_chunker(
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Chunker:
    """Return the chunker for `strategy` (default: settings.chunking_strategy).

    IN:  strategy name, max chunk size in characters, overlap in characters.
         Any argument left as None falls back to `app.config.get_settings()`.
    OUT: an object with a `.chunk(document) -> list[Chunk]` method.

    TODO(milestone 3):
      - Write at least one class with a `chunk` method (e.g. `FixedSizeChunker`).
      - Keep a dict of strategy name -> class, and look `strategy` up in it here.
      - Raise ValueError for an unknown strategy name.
    """
    raise NotImplementedError("TODO(milestone 3): implement get_chunker in app/vectordb/chunking.py")
