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

import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from app.config import get_settings
from app.models import Chunk, RetrievedChunk

Matrix = NDArray[np.float32]


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


INDEX_FILE = "index.json"  # chunk data + which vectors file belongs to it
VECTORS_PREFIX = "vectors-"  # vectors-<id>.npy, one float32 row per chunk
FORMAT_VERSION = 1


class StoreError(ValueError):
    """The files under persist_dir are unreadable or inconsistent."""


def _normalise(matrix: Matrix) -> Matrix:
    """Scale every row to length 1, so a dot product is a cosine similarity. All-zero rows stay zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


class NumpyVectorStore:
    """Chunks in `index.json`, vectors in a `.npy` matrix, searched by brute-force cosine similarity.

    - **Search:** vectors are stored normalised, so scoring is one matrix product. Exact (no approximation),
      and fast for thousands of chunks. For millions, switch to a real vector database.
    - **Persistence:** every change writes a *new* vectors file, then atomically swaps `index.json` to point at
      it. A crash can leave an unused vectors file behind, but never a half-written or mismatched store.
    - **Sharing:** if another process (for example `python -m app.ingest`) changes the files, this store
      reloads them on its next call, so a running server sees newly indexed documents without a restart.
    """

    def __init__(self, persist_dir: str | Path) -> None:
        self.directory = Path(persist_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._chunks: list[Chunk] = []
        self._matrix: Matrix = np.empty((0, 0), dtype=np.float32)
        self._dimension: int | None = None
        self._signature: tuple[int, int] | None = None
        self._load()

    # ------------------------------------------------------------------ disk
    @property
    def _index_path(self) -> Path:
        return self.directory / INDEX_FILE

    def _file_signature(self) -> tuple[int, int] | None:
        try:
            stat = self._index_path.stat()
        except FileNotFoundError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def _load(self) -> None:
        self._signature = self._file_signature()
        if self._signature is None:
            self._chunks, self._matrix, self._dimension = [], np.empty((0, 0), dtype=np.float32), None
            return
        try:
            index = json.loads(self._index_path.read_text(encoding="utf-8"))
            chunks = [Chunk.model_validate(item) for item in index["chunks"]]
            dimension = index["dimension"]
            vectors_file = index["vectors_file"]
            if vectors_file is None:
                matrix = np.empty((0, 0), dtype=np.float32)
            else:
                matrix = np.load(self.directory / vectors_file, allow_pickle=False)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise StoreError(f"Cannot read the vector store in {self.directory}: {exc}") from exc
        if chunks and matrix.shape != (len(chunks), dimension):
            raise StoreError(
                f"Vector store in {self.directory} is inconsistent: {len(chunks)} chunks, matrix {matrix.shape}"
            )
        self._chunks, self._matrix, self._dimension = chunks, matrix.astype(np.float32, copy=False), dimension

    def _refresh(self) -> None:
        """Reload when another process changed the files since we last looked."""
        if self._file_signature() != self._signature:
            self._load()

    def _atomic_write(self, path: Path, write: Any, suffix: str) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as handle:
                write(handle)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _save(self) -> None:
        vectors_file: str | None = None
        if self._chunks:
            vectors_file = f"{VECTORS_PREFIX}{uuid.uuid4().hex}.npy"
            self._atomic_write(self.directory / vectors_file, lambda fh: np.save(fh, self._matrix), ".tmp")
        index = {
            "version": FORMAT_VERSION,
            "dimension": self._dimension,
            "vectors_file": vectors_file,
            "chunks": [chunk.model_dump() for chunk in self._chunks],
        }
        payload = json.dumps(index, ensure_ascii=False).encode("utf-8")
        self._atomic_write(self._index_path, lambda fh: fh.write(payload), ".tmp")  # the swap: new data goes live
        self._signature = self._file_signature()
        for old in self.directory.glob(f"{VECTORS_PREFIX}*.npy"):  # now unused
            if old.name != vectors_file:
                old.unlink(missing_ok=True)

    # ------------------------------------------------------------------ API
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            return
        new = np.asarray(embeddings, dtype=np.float32)
        if new.ndim != 2 or new.shape[1] == 0:
            raise ValueError("embeddings must be a list of equal-length, non-empty vectors")
        with self._lock:
            self._refresh()
            if self._dimension is not None and new.shape[1] != self._dimension:
                raise ValueError(
                    f"Embeddings have {new.shape[1]} dimensions but this store holds {self._dimension}. "
                    f"If you changed EMBEDDING_MODEL, delete {self.directory} and re-index."
                )
            latest = {chunk.id: i for i, chunk in enumerate(chunks)}  # the same id twice in one call: last wins
            keep = sorted(latest.values())
            chunks, new = [chunks[i] for i in keep], _normalise(new[keep])

            positions = {chunk.id: i for i, chunk in enumerate(self._chunks)}
            if self._chunks:
                matrix, stored = self._matrix.copy(), list(self._chunks)
            else:
                matrix, stored = np.empty((0, new.shape[1]), dtype=np.float32), []
            appended: list[int] = []
            for row, chunk in enumerate(chunks):
                if chunk.id in positions:  # upsert: replace in place, keep the position
                    stored[positions[chunk.id]] = chunk
                    matrix[positions[chunk.id]] = new[row]
                else:
                    appended.append(row)
            self._chunks = stored + [chunks[row] for row in appended]
            self._matrix = np.vstack([matrix, new[appended]]) if appended else matrix
            self._dimension = new.shape[1]
            self._save()

    def search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        with self._lock:
            self._refresh()
            if k <= 0 or not self._chunks:
                return []
            query = np.asarray(query_embedding, dtype=np.float32)
            if query.ndim != 1 or query.shape[0] != self._dimension:
                raise ValueError(
                    f"Query has {query.shape} values but the store holds {self._dimension}-dimensional vectors"
                )
            scores = self._matrix @ _normalise(query[None, :])[0]
            order = np.argsort(-scores, kind="stable")[:k]  # stable: equal scores keep insertion order
            return [RetrievedChunk(chunk=self._chunks[i], score=float(scores[i])) for i in order]

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            self._refresh()
            keep = [i for i, chunk in enumerate(self._chunks) if chunk.doc_id != doc_id]
            removed = len(self._chunks) - len(keep)
            if removed == 0:
                return 0
            self._chunks = [self._chunks[i] for i in keep]
            self._matrix = self._matrix[keep]
            if not self._chunks:
                self._matrix, self._dimension = np.empty((0, 0), dtype=np.float32), None  # empty: any size may follow
            self._save()
            return removed

    def count(self) -> int:
        with self._lock:
            self._refresh()
            return len(self._chunks)


def get_vector_store(persist_dir: str | Path | None = None) -> VectorStore:
    """Open (or create) the vector store saved in `persist_dir` (default: settings.vector_dir).

    Raises StoreError (a ValueError) if the files there are corrupt, instead of silently overwriting them.
    """
    return NumpyVectorStore(persist_dir if persist_dir is not None else get_settings().vector_dir)
