"""Milestone 5 unit tests — the NumPy + JSON vector store. No API key, model or network needed."""

from __future__ import annotations

import json
import random
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from app.config import get_settings
from app.models import Chunk
from app.vectordb.store import INDEX_FILE, VECTORS_PREFIX, NumpyVectorStore, StoreError, get_vector_store
from tests.fakes import HashEmbedder, ListStore, make_chunk

EMBEDDER = HashEmbedder()
TEXTS = ["annual leave days", "expense receipts", "on-call stipend", "object storage price"]


def chunks_for(texts: list[str], doc_id: str = "doc-a") -> list[Chunk]:
    return [make_chunk(t, position=i, doc_id=doc_id) for i, t in enumerate(texts)]


def filled(path: Path, texts: list[str] = TEXTS, doc_id: str = "doc-a") -> NumpyVectorStore:
    store = NumpyVectorStore(path)
    store.add(chunks_for(texts, doc_id), EMBEDDER.embed_documents(texts))
    return store


def vector_files(path: Path) -> list[Path]:
    return sorted(path.glob(f"{VECTORS_PREFIX}*.npy"))


# --------------------------------------------------------------------------- search


def test_search_ranks_by_similarity_and_returns_full_chunks(tmp_path: Path) -> None:
    store = filled(tmp_path)
    hits = store.search(EMBEDDER.embed_query("expense receipts"), k=4)
    assert [h.chunk.text for h in hits][0] == "expense receipts"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert hits[0].score == pytest.approx(1.0, abs=1e-5), "a chunk's own vector is a perfect match"
    assert hits[0].chunk == chunks_for(TEXTS)[1], "every field of the Chunk comes back"


@pytest.mark.parametrize("k,expected", [(1, 1), (3, 3), (4, 4), (50, 4), (0, 0), (-2, 0)])
def test_search_returns_at_most_k(tmp_path: Path, k: int, expected: int) -> None:
    assert len(filled(tmp_path).search(EMBEDDER.embed_query("leave"), k)) == expected


def test_search_on_an_empty_store_returns_nothing(tmp_path: Path) -> None:
    assert NumpyVectorStore(tmp_path).search([0.1] * 256, k=3) == []


def test_scores_are_cosine_similarity_regardless_of_vector_length(tmp_path: Path) -> None:
    store = NumpyVectorStore(tmp_path)
    store.add(chunks_for(["x", "y"]), [[10.0, 0.0], [0.0, 0.001]])  # very different lengths, same directions
    hits = store.search([3.0, 4.0], k=2)  # direction (0.6, 0.8)
    scores = {h.chunk.text: h.score for h in hits}
    assert scores["x"] == pytest.approx(0.6, abs=1e-5)
    assert scores["y"] == pytest.approx(0.8, abs=1e-5)
    assert [h.chunk.text for h in hits] == ["y", "x"]


def test_zero_vectors_do_not_crash_or_produce_nan(tmp_path: Path) -> None:
    store = NumpyVectorStore(tmp_path)
    store.add(chunks_for(["zero", "one"]), [[0.0, 0.0], [1.0, 0.0]])
    assert [h.score for h in store.search([1.0, 0.0], 2)] == [pytest.approx(1.0), 0.0]
    assert [h.score for h in store.search([0.0, 0.0], 2)] == [0.0, 0.0], "a zero query matches nothing, no NaN"


def test_equal_scores_keep_insertion_order(tmp_path: Path) -> None:
    store = NumpyVectorStore(tmp_path)
    store.add(chunks_for(["a", "b", "c"]), [[1.0, 0.0]] * 3)
    assert [h.chunk.text for h in store.search([1.0, 0.0], 3)] == ["a", "b", "c"]


def test_matches_the_brute_force_reference_store(tmp_path: Path) -> None:
    rng = random.Random(7)
    texts = [
        " ".join(rng.choices(["leave", "expense", "storage", "oncall", "travel", "policy", "api"], k=6))
        for _ in range(60)
    ]
    reference, store = ListStore(), NumpyVectorStore(tmp_path)
    chunks, vectors = chunks_for(texts), EMBEDDER.embed_documents(texts)
    reference.add(chunks, vectors)
    store.add(chunks, vectors)
    for query in ("leave policy", "travel expense", "api storage oncall", "policy"):
        vector = EMBEDDER.embed_query(query)
        want = {h.chunk.id: h.score for h in reference.search(vector, 10)}
        got = {h.chunk.id: h.score for h in store.search(vector, 10)}
        assert got.keys() == want.keys() or [round(s, 5) for s in sorted(got.values())] == [
            round(s, 5) for s in sorted(want.values())
        ]
        for chunk_id, score in got.items():
            assert score == pytest.approx(want.get(chunk_id, score), abs=1e-5)


# --------------------------------------------------------------------------- add / upsert


def test_adding_the_same_id_replaces_text_and_vector_without_duplicating(tmp_path: Path) -> None:
    store = filled(tmp_path)
    replacement = make_chunk(
        "completely new wording about parking", position=1, doc_id="doc-a"
    )  # same id as "expense receipts"
    store.add([replacement], EMBEDDER.embed_documents([replacement.text]))
    assert store.count() == 4
    assert store.search(EMBEDDER.embed_query("parking wording"), 1)[0].chunk.text == replacement.text
    assert "expense receipts" not in [h.chunk.text for h in store.search(EMBEDDER.embed_query("expense receipts"), 4)]


def test_upsert_keeps_the_original_position_in_tie_order(tmp_path: Path) -> None:
    store = NumpyVectorStore(tmp_path)
    store.add(chunks_for(["a", "b", "c"]), [[1.0, 0.0]] * 3)
    store.add([make_chunk("b2", position=1, doc_id="doc-a")], [[1.0, 0.0]])
    assert [h.chunk.text for h in store.search([1.0, 0.0], 3)] == ["a", "b2", "c"]


def test_same_id_twice_in_one_call_keeps_the_last(tmp_path: Path) -> None:
    store = NumpyVectorStore(tmp_path)
    first, second = make_chunk("first", position=0), make_chunk("second", position=0)
    store.add([first, second], [[1.0, 0.0], [0.0, 1.0]])
    assert store.count() == 1
    assert store.search([0.0, 1.0], 1)[0].chunk.text == "second"


def test_add_nothing_is_a_no_op(tmp_path: Path) -> None:
    store = NumpyVectorStore(tmp_path)
    store.add([], [])
    assert store.count() == 0 and not (tmp_path / INDEX_FILE).exists()


def test_mismatched_lengths_are_rejected_and_change_nothing(tmp_path: Path) -> None:
    store = filled(tmp_path)
    with pytest.raises(ValueError, match="3 chunks but 2 embeddings"):
        store.add(chunks_for(["a", "b", "c"], "doc-b"), [[0.0] * 256] * 2)
    assert store.count() == 4


def test_ragged_vectors_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        NumpyVectorStore(tmp_path).add(chunks_for(["a", "b"]), [[1.0, 2.0], [1.0]])


def test_changing_embedding_size_gives_a_helpful_error(tmp_path: Path) -> None:
    store = filled(tmp_path)  # 256 dimensions
    with pytest.raises(ValueError, match=r"holds 256.*re-index"):
        store.add(chunks_for(["z"], "doc-b"), [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="256-dimensional"):
        store.search([1.0, 2.0, 3.0], 1)
    assert store.count() == 4


def test_a_new_embedding_size_is_allowed_once_the_store_is_empty(tmp_path: Path) -> None:
    store = filled(tmp_path)
    store.delete_document("doc-a")
    store.add(chunks_for(["z"], "doc-b"), [[1.0, 2.0, 3.0]])
    assert store.count() == 1 and len(store.search([1.0, 2.0, 3.0], 1)) == 1


# --------------------------------------------------------------------------- delete


def test_delete_document_removes_only_that_document(tmp_path: Path) -> None:
    store = filled(tmp_path, TEXTS[:2], "doc-a")
    store.add(chunks_for(TEXTS[2:], "doc-b"), EMBEDDER.embed_documents(TEXTS[2:]))
    assert store.delete_document("doc-a") == 2
    assert store.count() == 2
    assert {h.chunk.doc_id for h in store.search(EMBEDDER.embed_query("leave"), 10)} == {"doc-b"}


def test_delete_unknown_document_returns_zero_and_writes_nothing(tmp_path: Path) -> None:
    store = filled(tmp_path)
    before = (tmp_path / INDEX_FILE).stat().st_mtime_ns
    assert store.delete_document("nope") == 0
    assert (tmp_path / INDEX_FILE).stat().st_mtime_ns == before


def test_deleting_everything_leaves_a_working_empty_store(tmp_path: Path) -> None:
    store = filled(tmp_path)
    assert store.delete_document("doc-a") == 4
    assert store.count() == 0 and store.search(EMBEDDER.embed_query("x"), 3) == []
    assert vector_files(tmp_path) == [], "no orphan vector files"
    assert NumpyVectorStore(tmp_path).count() == 0


# --------------------------------------------------------------------------- persistence


def test_data_survives_reopening_with_every_field(tmp_path: Path) -> None:
    chunk = Chunk(id="d:0", doc_id="d", text="Ünïcode — 年次休暇", source="s.md", position=0, metadata={"k": [1, "v"]})
    NumpyVectorStore(tmp_path).add([chunk], [[0.5, 0.5]])
    reopened = NumpyVectorStore(tmp_path)
    assert reopened.count() == 1
    assert reopened.search([0.5, 0.5], 1)[0].chunk == chunk


def test_only_one_vectors_file_is_kept_after_many_changes(tmp_path: Path) -> None:
    store = NumpyVectorStore(tmp_path)
    for i in range(6):
        store.add(chunks_for([f"t{i}"], f"doc-{i}"), [[1.0, float(i)]])
    store.delete_document("doc-0")
    assert len(vector_files(tmp_path)) == 1
    assert not list(tmp_path.glob("*.tmp")), "no temp files left behind"


def test_a_crash_leaving_stray_files_does_not_break_the_store(tmp_path: Path) -> None:
    """Simulate dying between 'write new vectors' and 'swap the index': an unused file is left behind."""
    store = filled(tmp_path)
    np.save(tmp_path / f"{VECTORS_PREFIX}deadbeef.npy", np.zeros((9, 256), dtype=np.float32))
    reopened = NumpyVectorStore(tmp_path)
    assert reopened.count() == 4, "the index still points at the good vectors"
    reopened.add(chunks_for(["more"], "doc-b"), EMBEDDER.embed_documents(["more"]))
    assert len(vector_files(tmp_path)) == 1, "the next save cleans the stray file up"
    assert store is not reopened


def test_vectors_file_is_not_pickled(tmp_path: Path) -> None:
    filled(tmp_path)
    (vectors,) = vector_files(tmp_path)
    assert np.load(vectors, allow_pickle=False).dtype == np.float32


def test_another_store_instance_sees_changes_made_after_it_opened(tmp_path: Path) -> None:
    """The web server holds a store open while `python -m app.ingest` indexes documents in another process."""
    server = NumpyVectorStore(tmp_path)
    assert server.count() == 0
    filled(tmp_path)  # a second instance, like the ingest command
    assert server.count() == 4
    assert server.search(EMBEDDER.embed_query("on-call stipend"), 1)[0].chunk.text == "on-call stipend"
    NumpyVectorStore(tmp_path).delete_document("doc-a")
    assert server.count() == 0


def test_writes_from_two_instances_do_not_overwrite_each_other(tmp_path: Path) -> None:
    first, second = NumpyVectorStore(tmp_path), NumpyVectorStore(tmp_path)
    first.add(chunks_for(["one"], "doc-1"), [[1.0, 0.0]])
    second.add(chunks_for(["two"], "doc-2"), [[0.0, 1.0]])  # must start from first's data, not an empty copy
    assert first.count() == second.count() == 2


# --------------------------------------------------------------------------- bad files


def test_corrupt_index_raises_instead_of_being_overwritten(tmp_path: Path) -> None:
    filled(tmp_path)
    (tmp_path / INDEX_FILE).write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(StoreError, match="Cannot read the vector store"):
        NumpyVectorStore(tmp_path)
    assert (tmp_path / INDEX_FILE).read_text() == "{ this is not json", "the bad file was left for the user to inspect"


def test_missing_vectors_file_raises(tmp_path: Path) -> None:
    filled(tmp_path)
    vector_files(tmp_path)[0].unlink()
    with pytest.raises(StoreError):
        NumpyVectorStore(tmp_path)


def test_matrix_that_does_not_match_the_chunks_raises(tmp_path: Path) -> None:
    filled(tmp_path)
    index = json.loads((tmp_path / INDEX_FILE).read_text())
    index["chunks"] = index["chunks"][:2]
    (tmp_path / INDEX_FILE).write_text(json.dumps(index))
    with pytest.raises(StoreError, match="inconsistent"):
        NumpyVectorStore(tmp_path)


# --------------------------------------------------------------------------- factory / threads


@pytest.fixture()
def fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_default_directory_comes_from_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fresh_settings: None
) -> None:
    target = tmp_path / "from-settings"
    monkeypatch.setenv("VECTOR_DIR", str(target))
    store = get_vector_store()
    store.add(chunks_for(["a"]), [[1.0, 0.0]])
    assert (target / INDEX_FILE).exists()


def test_persist_dir_may_be_a_string_and_is_created(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b"
    get_vector_store(persist_dir=str(deep)).add(chunks_for(["a"]), [[1.0, 0.0]])
    assert get_vector_store(persist_dir=deep).count() == 1


def test_concurrent_adds_from_many_threads_lose_nothing(tmp_path: Path) -> None:
    store = NumpyVectorStore(tmp_path)
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for j in range(5):
                doc = f"doc-{n}-{j}"
                store.add(chunks_for([f"text {n} {j}"], doc), [[float(n + 1), float(j + 1)]])
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert store.count() == 40 == NumpyVectorStore(tmp_path).count()
    assert len(vector_files(tmp_path)) == 1
