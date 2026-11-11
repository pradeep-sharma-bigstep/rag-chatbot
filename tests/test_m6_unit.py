"""Milestone 6 unit tests — loading, ingesting one file, ingesting a folder, uploads and the CLI.

Uses the project's own chunker and vector store with the fake hash embedder, so no model download
or API key is needed (except the one CLI test, which runs the real command and the real model).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT, Settings
from app.deps import Components, get_components
from app.ingest import pipeline
from app.ingest.pipeline import ingest_directory, ingest_file, load_document
from app.main import create_app
from app.models import Chunk, RetrievedChunk
from app.vectordb.chunking import get_chunker
from app.vectordb.store import NumpyVectorStore
from tests.conftest import SAMPLE_DOCS
from tests.fakes import HashEmbedder, ListStore

LEAVE = SAMPLE_DOCS / "leave_policy.md"


def parts(store: object | None = None, size: int = 300) -> dict[str, object]:
    return {
        "chunker": get_chunker("recursive", size, 0),
        "embedder": HashEmbedder(),
        "store": store if store is not None else ListStore(),
    }


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def texts_in(store: ListStore) -> list[str]:
    return [chunk.text for chunk, _ in store.items.values()]


# --------------------------------------------------------------------------- load_document


def test_load_document_fields() -> None:
    doc = load_document(LEAVE)
    assert doc.source == "leave_policy.md"
    assert doc.text == LEAVE.read_text(encoding="utf-8")
    assert doc.id.startswith("doc-") and doc.metadata == {"suffix": ".md"}


def test_id_depends_only_on_the_file_name(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = write(tmp_path / "a", "notes.md", "version one")
    other_folder = write(tmp_path / "b", "notes.md", "totally different content")
    other_name = write(tmp_path, "notes2.md", "version one")
    assert load_document(first).id == load_document(other_folder).id, "same name, other folder/content: same id"
    assert load_document(first).id != load_document(other_name).id
    first.write_text("edited", encoding="utf-8")
    assert load_document(first).id == load_document(other_folder).id, "editing the file keeps its id"


def test_all_sample_documents_have_distinct_ids() -> None:
    ids = [load_document(p).id for p in sorted(SAMPLE_DOCS.glob("*.*"))]
    assert len(ids) == len(set(ids)) == 4


@pytest.mark.parametrize("name", ["a.md", "a.txt", "A.MD", "b.Txt"])
def test_supported_suffixes_are_case_insensitive(tmp_path: Path, name: str) -> None:
    assert load_document(write(tmp_path, name, "hello")).text == "hello"


@pytest.mark.parametrize("name", ["x.exe", "x.pdf", "x", "x.md.bak", ".md"])
def test_unsupported_files_raise_value_error(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(b"data")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(path)


def test_utf8_bom_is_dropped_and_windows_newlines_are_normalised(tmp_path: Path) -> None:
    path = tmp_path / "w.txt"
    path.write_bytes(b"\xef\xbb\xbfLine one\r\nLine two\r\n")
    assert load_document(path).text == "Line one\nLine two\n"


def test_unicode_text_is_kept(tmp_path: Path) -> None:
    text = "Café — 年次休暇は24日です 😀"
    assert load_document(write(tmp_path, "u.md", text)).text == text


def test_invalid_utf8_raises_a_clear_value_error(tmp_path: Path) -> None:
    path = tmp_path / "latin1.txt"
    path.write_bytes("café".encode("latin-1"))
    with pytest.raises(ValueError, match=r"'latin1.txt' is not valid UTF-8"):
        load_document(path)


def test_missing_file_and_folder_are_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "ghost.md")
    folder = tmp_path / "folder.md"
    folder.mkdir()
    with pytest.raises(ValueError, match="not a file"):
        load_document(folder)


# --------------------------------------------------------------------------- ingest_file


def test_ingest_file_returns_the_number_of_chunks_stored(tmp_path: Path) -> None:
    store = ListStore()
    added = ingest_file(LEAVE, **parts(store))  # type: ignore[arg-type]
    expected = get_chunker("recursive", 300, 0).chunk(load_document(LEAVE))
    assert added == len(expected) == store.count() > 1
    assert sorted(texts_in(store)) == sorted(c.text for c in expected)


def test_stored_chunks_carry_the_document_id_and_file_name() -> None:
    store = ListStore()
    ingest_file(LEAVE, **parts(store))  # type: ignore[arg-type]
    doc = load_document(LEAVE)
    assert all(c.doc_id == doc.id and c.source == "leave_policy.md" for c, _ in store.items.values())
    assert all(chunk.id.startswith(f"{doc.id}:") for chunk, _ in store.items.values())


def test_ingesting_twice_does_not_duplicate() -> None:
    store = ListStore()
    first = ingest_file(LEAVE, **parts(store))  # type: ignore[arg-type]
    second = ingest_file(LEAVE, **parts(store))  # type: ignore[arg-type]
    assert first == second == store.count()


def test_a_shorter_new_version_leaves_no_stale_chunks(tmp_path: Path) -> None:
    store = NumpyVectorStore(tmp_path / "vs")
    docs = tmp_path / "docs"
    docs.mkdir()
    path = write(docs, "policy.md", "\n\n".join(f"Paragraph number {i} with some words in it." for i in range(30)))
    long_count = ingest_file(path, **parts(store, size=120))  # type: ignore[arg-type]
    path.write_text("Only one short paragraph now.", encoding="utf-8")
    short_count = ingest_file(path, **parts(store, size=120))  # type: ignore[arg-type]
    assert long_count > 5 and short_count == 1
    assert store.count() == 1, "old chunks 1..n must be gone, not just overwritten at position 0"
    assert (
        HashEmbedder().embed_query("Paragraph number 12")
        and store.search(HashEmbedder().embed_query("Paragraph number 12"), 5)[0].chunk.text
        == "Only one short paragraph now."
    )


def test_a_file_that_becomes_empty_is_removed_from_the_index(tmp_path: Path) -> None:
    store = ListStore()
    path = write(tmp_path, "doc.md", "Some real content to index here.")
    assert ingest_file(path, **parts(store)) == 1  # type: ignore[arg-type]
    path.write_text("   \n\n  ", encoding="utf-8")
    assert ingest_file(path, **parts(store)) == 0  # type: ignore[arg-type]
    assert store.count() == 0


def test_a_blank_file_stores_nothing_and_never_calls_the_embedder(tmp_path: Path) -> None:
    class Spy(HashEmbedder):
        calls = 0

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            Spy.calls += 1
            return super().embed_documents(texts)

    store = ListStore()
    result = ingest_file(
        write(tmp_path, "blank.txt", "  \n"), chunker=get_chunker("fixed", 100, 0), embedder=Spy(), store=store
    )
    assert result == 0 and store.count() == 0 and Spy.calls == 0


def test_other_documents_are_untouched_when_one_is_reingested() -> None:
    store = ListStore()
    ingest_file(LEAVE, **parts(store))  # type: ignore[arg-type]
    ingest_file(SAMPLE_DOCS / "expense_policy.md", **parts(store))  # type: ignore[arg-type]
    before = store.count()
    ingest_file(LEAVE, **parts(store))  # type: ignore[arg-type]
    assert store.count() == before
    assert {c.source for c, _ in store.items.values()} == {"leave_policy.md", "expense_policy.md"}


def test_embedder_failure_keeps_the_previous_version_searchable(tmp_path: Path) -> None:
    class Broken(HashEmbedder):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("model crashed")

    store = ListStore()
    path = write(tmp_path, "doc.md", "Original text that should stay indexed.")
    ingest_file(path, **parts(store))  # type: ignore[arg-type]
    path.write_text("New text that never gets indexed.", encoding="utf-8")
    with pytest.raises(RuntimeError, match="model crashed"):
        ingest_file(path, chunker=get_chunker("fixed", 300, 0), embedder=Broken(), store=store)
    assert texts_in(store) == ["Original text that should stay indexed."]


def test_wrong_number_of_vectors_is_rejected_before_touching_the_store(tmp_path: Path) -> None:
    class Short(HashEmbedder):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return super().embed_documents(texts)[:-1]

    store = ListStore()
    path = write(tmp_path, "doc.md", "Para one.\n\nPara two.\n\nPara three.")
    with pytest.raises(ValueError, match="vectors for"):
        ingest_file(path, chunker=get_chunker("recursive", 12, 0), embedder=Short(), store=store)
    assert store.count() == 0


def test_ingest_file_uses_the_components_it_is_given(tmp_path: Path) -> None:
    calls: list[str] = []

    class Chunky:
        def chunk(self, document):  # type: ignore[no-untyped-def]
            calls.append("chunk")
            return [Chunk(id=f"{document.id}:0", doc_id=document.id, text="X", source=document.source, position=0)]

    class Embed(HashEmbedder):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            calls.append("embed")
            return super().embed_documents(texts)

    class Store(ListStore):
        def delete_document(self, doc_id: str) -> int:
            calls.append("delete")
            return super().delete_document(doc_id)

        def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
            calls.append("add")
            super().add(chunks, embeddings)

    ingest_file(write(tmp_path, "d.md", "anything"), chunker=Chunky(), embedder=Embed(), store=Store())
    assert calls == ["chunk", "embed", "delete", "add"], "embed first, then replace the old copy"


# --------------------------------------------------------------------------- ingest_directory


def test_ingest_directory_total_equals_store_count() -> None:
    store = ListStore()
    total = ingest_directory(SAMPLE_DOCS, **parts(store))  # type: ignore[arg-type]
    assert total == store.count() > 10
    assert {c.source for c, _ in store.items.values()} == {p.name for p in SAMPLE_DOCS.glob("*.*")}


def test_ingest_directory_twice_is_idempotent() -> None:
    store = ListStore()
    first = ingest_directory(SAMPLE_DOCS, **parts(store))  # type: ignore[arg-type]
    second = ingest_directory(SAMPLE_DOCS, **parts(store))  # type: ignore[arg-type]
    assert first == second == store.count()


def test_files_are_processed_sorted_by_name_and_only_supported_ones(tmp_path: Path) -> None:
    for name in ("b.md", "a.txt", "C.MD", "image.png", "notes.docx", ".DS_Store"):
        (tmp_path / name).write_text("hello world content", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    write(tmp_path / "sub", "nested.md", "should be ignored")
    order: list[str] = []

    class Spy(HashEmbedder):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            order.append(texts[0])
            return super().embed_documents(texts)

    store = ListStore()
    total = ingest_directory(tmp_path, chunker=get_chunker("fixed", 100, 0), embedder=Spy(), store=store)
    assert total == 3 == store.count()
    assert sorted(c.source for c, _ in store.items.values()) == ["C.MD", "a.txt", "b.md"]
    assert [c.source for c, _ in store.items.values()] == ["C.MD", "a.txt", "b.md"], (
        "sorted(): 'C.MD' < 'a.txt' < 'b.md'"
    )


def test_empty_directory_gives_zero(tmp_path: Path) -> None:
    assert ingest_directory(tmp_path, **parts()) == 0  # type: ignore[arg-type]


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        ingest_directory(tmp_path / "nope", **parts())  # type: ignore[arg-type]
    file = write(tmp_path, "f.md", "x")
    with pytest.raises(NotADirectoryError):
        ingest_directory(file, **parts())  # type: ignore[arg-type]


def test_one_bad_file_is_skipped_with_a_warning_and_the_rest_are_indexed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write(tmp_path, "a_good.md", "Good document about leave.")
    (tmp_path / "b_bad.txt").write_bytes("café".encode("latin-1"))
    write(tmp_path, "c_good.md", "Another good document about expenses.")
    store = ListStore()
    with caplog.at_level(logging.WARNING, logger=pipeline.logger.name):
        total = ingest_directory(tmp_path, **parts(store))  # type: ignore[arg-type]
    assert total == 2 and {c.source for c, _ in store.items.values()} == {"a_good.md", "c_good.md"}
    assert "b_bad.txt" in caplog.text and "UTF-8" in caplog.text


def test_embedder_errors_are_not_swallowed_by_the_directory_loop(tmp_path: Path) -> None:
    class Broken(HashEmbedder):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("model crashed")

    write(tmp_path, "a.md", "content")
    with pytest.raises(RuntimeError, match="model crashed"):
        ingest_directory(tmp_path, chunker=get_chunker("fixed", 100, 0), embedder=Broken(), store=ListStore())


# --------------------------------------------------------------------------- end to end with the real store


def test_ingested_documents_can_be_searched_back(tmp_path: Path) -> None:
    store = NumpyVectorStore(tmp_path)
    embedder = HashEmbedder()
    ingest_directory(SAMPLE_DOCS, chunker=get_chunker("recursive", 300, 50), embedder=embedder, store=store)
    for question, source in [
        ("annual leave carried over to the next year", "leave_policy.md"),
        ("expense receipts reimbursement", "expense_policy.md"),
    ]:
        hit: RetrievedChunk = store.search(embedder.embed_query(question), 1)[0]
        assert hit.chunk.source == source, question
    reopened = NumpyVectorStore(tmp_path)
    assert reopened.count() == store.count(), "the index is persisted"


# --------------------------------------------------------------------------- uploads through the web app


@pytest.fixture()
def indexing_client(settings: Settings, tmp_path: Path) -> tuple[TestClient, NumpyVectorStore]:
    store = NumpyVectorStore(tmp_path / "vs")
    components = Components(
        engine=None,  # type: ignore[arg-type]  # not used by the upload endpoints
        chunker=get_chunker("recursive", 300, 0),
        embedder=HashEmbedder(),
        store=store,
    )
    app = create_app(settings)
    app.dependency_overrides[get_components] = lambda: components
    return TestClient(app), store


def test_upload_is_indexed(indexing_client: tuple[TestClient, NumpyVectorStore]) -> None:
    client, store = indexing_client
    body = b"# Parking\n\nVisitors may park in lot B for up to 4 hours.\n\nEmployees need a permit for lot A."
    response = client.post("/api/documents", files={"file": ("parking.md", body, "text/markdown")})
    assert response.status_code == 201
    payload = response.json()
    assert payload["indexed"] is True and payload["chunks_indexed"] == store.count() >= 1
    hit = store.search(HashEmbedder().embed_query("where can visitors park"), 1)[0]
    assert hit.chunk.source == "parking.md"


def test_uploading_the_same_name_again_replaces_the_old_chunks(
    indexing_client: tuple[TestClient, NumpyVectorStore],
) -> None:
    client, store = indexing_client
    client.post("/api/documents", files={"file": ("p.md", b"Old text.\n\n" * 40, "text/markdown")})
    client.post("/api/documents", files={"file": ("p.md", b"Brand new short text.", "text/markdown")})
    assert store.count() == 1
    assert store.search(HashEmbedder().embed_query("brand new"), 1)[0].chunk.text == "Brand new short text."


def test_browser_upload_form_reports_indexing(indexing_client: tuple[TestClient, NumpyVectorStore]) -> None:
    client, store = indexing_client
    response = client.post("/ui/documents", files={"file": ("faq.txt", b"Q: Where?\nA: Here.", "text/plain")})
    assert response.status_code == 200 and "indexed" in response.text.lower()
    assert store.count() == 1


# --------------------------------------------------------------------------- the real command


def test_cli_indexes_documents_and_is_idempotent(tmp_path: Path) -> None:
    docs, vectors = tmp_path / "docs", tmp_path / "vectors"
    docs.mkdir()
    for path in SAMPLE_DOCS.glob("*.*"):
        (docs / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    env = {**os.environ, "DOCUMENTS_DIR": str(docs), "VECTOR_DIR": str(vectors), "CHUNKING_STRATEGY": "recursive"}

    def run() -> str:
        done = subprocess.run(
            [sys.executable, "-m", "app.ingest"], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=300
        )
        if done.returncode != 0 and "Unsupported EMBEDDING_MODEL" in done.stderr + done.stdout:
            pytest.skip("embedding model not available")
        assert done.returncode == 0, done.stderr[-800:]
        return done.stdout.strip().splitlines()[-1]

    first = run()
    count = NumpyVectorStore(vectors).count()
    assert first.startswith(f"Indexed {count} chunks") and count > 10
    run()
    assert NumpyVectorStore(vectors).count() == count, "running the command twice must not duplicate"
    assert {h.chunk.source for h in NumpyVectorStore(vectors).search([1.0] + [0.0] * 383, 50)} == {
        p.name for p in docs.iterdir()
    }


def test_settings_replace_still_works_for_other_tests(settings: Settings) -> None:
    """Guards the fixture used above: `replace` on the frozen Settings must keep working."""
    assert replace(settings, top_k=9).top_k == 9
