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

import re
from typing import Protocol

from app.config import get_settings
from app.models import Chunk, Document


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]:
        """IN: one Document.  OUT: its Chunks, in order."""
        ...


# --------------------------------------------------------------------------- shared helpers


def _validate(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(f"chunk_overlap must be between 0 and chunk_size - 1, got {chunk_overlap} (size {chunk_size})")


def _to_chunks(document: Document, texts: list[str]) -> list[Chunk]:
    """Turn raw pieces into Chunk objects: drop blanks, trim, number 0..n-1, build stable ids."""
    pieces = [text.strip() for text in texts]
    return [
        Chunk(
            id=f"{document.id}:{position}",
            doc_id=document.id,
            text=piece,
            source=document.source,
            position=position,
            metadata=dict(document.metadata),
        )
        for position, piece in enumerate(p for p in pieces if p)
    ]


# --------------------------------------------------------------------------- strategy: fixed


class FixedSizeChunker:
    """Sliding window of `chunk_size` characters that steps forward by `chunk_size - overlap`.

    Windows are moved back to the last whitespace so words are not cut in half. Only a single
    word longer than the whole window is split (there is no other way to respect `chunk_size`).
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        _validate(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _window_end(self, text: str, start: int) -> int:
        end = min(start + self.chunk_size, len(text))
        if end < len(text) and not text[end].isspace():  # the window would cut a word in half
            cut = end
            while cut > start and not text[cut - 1].isspace():
                cut -= 1
            if cut > start:
                return cut
        return end

    def _next_start(self, text: str, start: int, end: int) -> int:
        candidate = end - self.chunk_overlap
        if candidate <= start:  # the window was short, so no room for overlap
            return end
        while candidate < end and not text[candidate - 1].isspace() and not text[candidate].isspace():
            candidate += 1  # begin the next window at a word boundary
        return candidate

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        pieces: list[str] = []
        start = 0
        while start < len(text):
            end = self._window_end(text, start)
            pieces.append(text[start:end])
            start = end if end >= len(text) else self._next_start(text, start, end)
        return _to_chunks(document, pieces)


# --------------------------------------------------------------------------- strategy: recursive


_WORD_AND_TRAILING_SPACE = re.compile(r"\S+\s*|\s+")  # splits at any whitespace (space, tab, \r ...)


class RecursiveChunker:
    """Split on the largest natural boundary that fits: paragraph, then line, then sentence, then word.

    Small pieces are packed together up to `chunk_size`. Chunks therefore start and end at natural
    boundaries. Overlap repeats the last whole pieces of the previous chunk (up to `chunk_overlap`).
    """

    SEPARATORS = ("\n\n", "\n", ". ", " ")

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        _validate(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _atoms(self, text: str, separators: tuple[str, ...]) -> list[str]:
        """Pieces of at most `chunk_size`; their concatenation is exactly `text` (separators are kept)."""
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:  # no boundary left: cut at the size limit
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        separator, rest = separators[0], separators[1:]
        if separator == " ":  # word level: any whitespace counts, not only the space character
            parts = _WORD_AND_TRAILING_SPACE.findall(text)
        else:
            parts = text.split(separator)
            parts = [part + separator for part in parts[:-1]] + parts[-1:]
        return [atom for part in parts if part for atom in self._atoms(part, rest)]

    def _carry(self, current: list[str], incoming: str) -> list[str]:
        """Trailing pieces of the chunk just finished that are reused to start the next one."""
        carried: list[str] = []
        total = 0
        for atom in reversed(current):
            if total + len(atom) > self.chunk_overlap:
                break
            carried.insert(0, atom)
            total += len(atom)
        while carried and total + len(incoming) > self.chunk_size:
            total -= len(carried.pop(0))
        return carried

    def chunk(self, document: Document) -> list[Chunk]:
        pieces: list[str] = []
        current: list[str] = []
        length = 0
        for atom in self._atoms(document.text, self.SEPARATORS):
            if current and length + len(atom) > self.chunk_size:
                pieces.append("".join(current))
                current = self._carry(current, atom)
                length = sum(len(a) for a in current)
            current.append(atom)
            length += len(atom)
        pieces.append("".join(current))
        return _to_chunks(document, pieces)


# --------------------------------------------------------------------------- factory

STRATEGIES: dict[str, type[FixedSizeChunker] | type[RecursiveChunker]] = {
    "fixed": FixedSizeChunker,
    "recursive": RecursiveChunker,
}


def get_chunker(
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Chunker:
    """Return the chunker for `strategy` (default: settings.chunking_strategy).

    IN:  strategy name, max chunk size in characters, overlap in characters.
         Any argument left as None falls back to `app.config.get_settings()`.
    OUT: an object with a `.chunk(document) -> list[Chunk]` method.

    Raises ValueError for an unknown strategy or an invalid size/overlap. An overlap that comes
    from the settings (not passed in) is capped at a quarter of the size, so asking for a small
    `chunk_size` alone never fails because of the default overlap.
    """
    settings = get_settings()
    name = (strategy if strategy is not None else settings.chunking_strategy).strip().lower()
    if name not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy {name!r}; expected one of {sorted(STRATEGIES)}")
    size = chunk_size if chunk_size is not None else settings.chunk_size
    if chunk_overlap is not None:
        overlap = chunk_overlap
    else:
        overlap = min(settings.chunk_overlap, size // 4) if size > 0 else settings.chunk_overlap
    return STRATEGIES[name](size, overlap)
