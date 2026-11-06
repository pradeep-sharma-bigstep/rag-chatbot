"""Milestone 3 unit tests — both chunking strategies against the eight rules, plus edge cases.

No API key or network needed. The fuzz test uses a fixed seed, so it is repeatable.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator

import pytest

from app.config import get_settings
from app.models import Chunk, Document
from app.vectordb.chunking import STRATEGIES, FixedSizeChunker, RecursiveChunker, get_chunker
from tests.conftest import SAMPLE_DOCS

STRATEGY_NAMES = sorted(STRATEGIES)


def make_doc(text: str, doc_id: str = "doc-x", source: str = "x.md") -> Document:
    return Document(id=doc_id, text=text, source=source)


def words(text: str) -> set[str]:
    return set(re.findall(r"\S+", text))


def assert_rules(doc: Document, chunks: list[Chunk], size: int) -> None:
    """The eight rules from the top of app/vectordb/chunking.py."""
    assert all(c.text.strip() for c in chunks), "no empty chunks"
    assert all(len(c.text) <= size for c in chunks), "respect chunk_size"
    assert [c.position for c in chunks] == list(range(len(chunks))), "positions 0..n-1"
    assert len({c.id for c in chunks}) == len(chunks), "unique ids"
    assert all(c.doc_id == doc.id and c.source == doc.source for c in chunks)
    long_words = {w for w in words(doc.text) if len(w) > size}  # unavoidably split to respect the size
    lost = words(doc.text) - long_words - set().union(*(words(c.text) for c in chunks), set())
    assert not lost, f"words lost: {sorted(lost)[:5]}"


@pytest.fixture()
def fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# --------------------------------------------------------------------------- the rules, on real documents


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
@pytest.mark.parametrize("size,overlap", [(120, 0), (300, 50), (300, 0), (800, 100), (5000, 100)])
@pytest.mark.parametrize("path", sorted(SAMPLE_DOCS.glob("*.*")), ids=lambda p: p.name)
def test_rules_hold_on_sample_documents(strategy: str, size: int, overlap: int, path) -> None:  # type: ignore[no-untyped-def]
    doc = make_doc(path.read_text(encoding="utf-8"), doc_id=f"doc-{path.stem}", source=path.name)
    chunks = get_chunker(strategy, size, overlap).chunk(doc)
    assert chunks
    assert_rules(doc, chunks, size)


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_deterministic(strategy: str) -> None:
    doc = make_doc((SAMPLE_DOCS / "leave_policy.md").read_text())
    first = get_chunker(strategy, 300, 50).chunk(doc)
    second = get_chunker(strategy, 300, 50).chunk(doc)
    assert first == second


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_smaller_size_gives_more_chunks(strategy: str) -> None:
    doc = make_doc((SAMPLE_DOCS / "expense_policy.md").read_text())
    counts = [len(get_chunker(strategy, size, 0).chunk(doc)) for size in (2000, 800, 300, 100)]
    assert counts == sorted(counts) and len(set(counts)) == len(counts), counts


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_short_document_is_one_chunk(strategy: str) -> None:
    chunks = get_chunker(strategy, 800, 100).chunk(make_doc("  Just one short line.  \n"))
    assert [c.text for c in chunks] == ["Just one short line."]
    assert chunks[0].id == "doc-x:0"


# --------------------------------------------------------------------------- edge cases


EDGE_CASES = {
    "empty": "",
    "spaces": "   ",
    "only newlines": "\n\n\n\t \n",
    "one huge word": "x" * 1000,
    "huge word amid text": "before " + "y" * 450 + " after " + "z" * 90,
    "no spaces, 2 paragraphs": ("a" * 200) + "\n\n" + ("b" * 200),
    "crlf line endings": "Title\r\n\r\nFirst paragraph here.\r\n\r\nSecond paragraph here.\r\n" * 20,
    "unicode": "Café résumé — 年次休暇は24日です。 naïve 😀 " * 40,
    "many blank lines": "a b c\n\n\n\n\n\n\n\n\n\nd e f\n\n\n\n" * 30,
    "trailing whitespace": "word " * 100 + "   \n\n   ",
    "word exactly chunk size": "w" * 50 + " " + "v" * 50 + " " + "u" * 50,
}


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
@pytest.mark.parametrize("name", EDGE_CASES)
def test_edge_cases(strategy: str, name: str) -> None:
    text = EDGE_CASES[name]
    doc = make_doc(text)
    chunks = get_chunker(strategy, 50, 10).chunk(doc)
    if not text.strip():
        assert chunks == []
    else:
        assert chunks
        assert_rules(doc, chunks, 50)


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_blank_document_gives_no_chunks(strategy: str) -> None:
    assert get_chunker(strategy, 100, 0).chunk(make_doc("  \n \t")) == []


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_huge_word_is_split_without_losing_characters(strategy: str) -> None:
    text = "q" * 1000
    chunks = get_chunker(strategy, 100, 0).chunk(make_doc(text))
    assert all(len(c.text) <= 100 for c in chunks)
    assert "".join(c.text for c in chunks) == text


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_chunk_size_of_one_still_terminates(strategy: str) -> None:
    doc = make_doc("ab cd\n\nef")
    chunks = get_chunker(strategy, 1, 0).chunk(doc)
    assert all(len(c.text) == 1 for c in chunks)
    assert "".join(c.text for c in chunks) == "abcdef"


# --------------------------------------------------------------------------- fuzz


def random_text(rng: random.Random) -> str:
    vocabulary = ["a", "bb", "ccc", "leave", "policy", "x" * 30, "résumé", "年次休暇", "end.", "ok."]
    separators = [" ", " ", " ", "\n", "\n\n", "  ", "\t", ". ", "\r\n"]
    parts: list[str] = []
    for _ in range(rng.randint(0, 150)):
        parts.append(rng.choice(vocabulary))
        parts.append(rng.choice(separators))
    return "".join(parts)


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_fuzz_rules_hold_for_random_documents(strategy: str) -> None:
    rng = random.Random(20261106)
    for _ in range(300):
        size = rng.randint(1, 200)
        overlap = rng.randint(0, size - 1)
        doc = make_doc(random_text(rng))
        chunks = get_chunker(strategy, size, overlap).chunk(doc)
        if not doc.text.strip():
            assert chunks == []
        else:
            assert chunks, (size, overlap, doc.text)
        assert_rules(doc, chunks, size)


# --------------------------------------------------------------------------- strategy behaviour


def test_fixed_overlap_repeats_text_between_neighbours() -> None:
    text = " ".join(f"word{i:03d}" for i in range(200))
    chunks = FixedSizeChunker(chunk_size=100, chunk_overlap=30).chunk(make_doc(text))
    assert len(chunks) > 3
    for left, right in zip(chunks, chunks[1:], strict=False):
        assert words(left.text) & words(right.text), "neighbouring chunks must share some words"
    no_overlap = FixedSizeChunker(chunk_size=100, chunk_overlap=0).chunk(make_doc(text))
    for left, right in zip(no_overlap, no_overlap[1:], strict=False):
        assert not words(left.text) & words(right.text)


def test_fixed_does_not_cut_words_in_half() -> None:
    text = " ".join(f"word{i:03d}" for i in range(200))
    chunks = FixedSizeChunker(chunk_size=100, chunk_overlap=30).chunk(make_doc(text))
    vocabulary = words(text)
    assert all(words(c.text) <= vocabulary for c in chunks)


def test_recursive_starts_chunks_at_paragraph_boundaries() -> None:
    paragraphs = [f"Paragraph {i}. " + "filler " * 12 for i in range(10)]
    text = "\n\n".join(p.strip() for p in paragraphs)
    chunks = RecursiveChunker(chunk_size=250, chunk_overlap=0).chunk(make_doc(text))
    assert len(chunks) > 1
    assert all(c.text.startswith("Paragraph") for c in chunks)
    assert all(c.text.endswith("filler") for c in chunks)


def test_recursive_overlap_repeats_whole_pieces() -> None:
    paragraphs = [f"Paragraph {i} " + "filler " * 5 for i in range(12)]
    text = "\n\n".join(p.strip() for p in paragraphs)
    chunks = RecursiveChunker(chunk_size=120, chunk_overlap=60).chunk(make_doc(text))
    shared = [
        set(re.findall(r"Paragraph \d+", a.text)) & set(re.findall(r"Paragraph \d+", b.text))
        for a, b in zip(chunks, chunks[1:], strict=False)
    ]
    assert any(shared), "a paragraph at the end of one chunk should reappear at the start of the next"


def test_chunk_carries_document_metadata_without_sharing_it() -> None:
    doc = Document(id="d", text="hello world", source="s.md", metadata={"author": "ops"})
    chunk = get_chunker("fixed", 50, 0).chunk(doc)[0]
    assert chunk.metadata == {"author": "ops"}
    chunk.metadata["author"] = "changed"
    assert doc.metadata == {"author": "ops"}


# --------------------------------------------------------------------------- the factory


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        get_chunker("semantic-magic", 100, 0)


@pytest.mark.parametrize("size,overlap", [(0, 0), (-5, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_size_or_overlap_raises(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        get_chunker("fixed", size, overlap)


def test_strategy_name_is_case_and_whitespace_insensitive() -> None:
    assert isinstance(get_chunker(" Recursive ", 100, 0), RecursiveChunker)


def test_defaults_come_from_settings(monkeypatch: pytest.MonkeyPatch, fresh_settings: None) -> None:
    monkeypatch.setenv("CHUNKING_STRATEGY", "recursive")
    monkeypatch.setenv("CHUNK_SIZE", "400")
    monkeypatch.setenv("CHUNK_OVERLAP", "40")
    chunker = get_chunker()
    assert isinstance(chunker, RecursiveChunker)
    assert (chunker.chunk_size, chunker.chunk_overlap) == (400, 40)


def test_explicit_arguments_beat_settings(monkeypatch: pytest.MonkeyPatch, fresh_settings: None) -> None:
    monkeypatch.setenv("CHUNKING_STRATEGY", "recursive")
    monkeypatch.setenv("CHUNK_SIZE", "400")
    chunker = get_chunker("fixed", 200, 0)
    assert isinstance(chunker, FixedSizeChunker)
    assert (chunker.chunk_size, chunker.chunk_overlap) == (200, 0), "an explicit overlap of 0 must be kept"


def test_default_overlap_is_capped_when_only_a_small_size_is_given(
    monkeypatch: pytest.MonkeyPatch, fresh_settings: None
) -> None:
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    chunker = get_chunker("fixed", chunk_size=60)  # 100 would be larger than the size itself
    assert isinstance(chunker, FixedSizeChunker)
    assert chunker.chunk_overlap == 15


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
@pytest.mark.parametrize("whitespace", ["\t", "\r\n", "\r", " ", "\n"])
def test_any_whitespace_separates_words_never_cuts_them(strategy: str, whitespace: str) -> None:
    """Regression: a tab used to be ignored by the recursive word split, cutting 'end..' in half."""
    text = whitespace.join(["end.", "end..", "alpha", "bravo"] * 10)
    doc = make_doc(text)
    chunks = get_chunker(strategy, 7, 0).chunk(doc)
    assert_rules(doc, chunks, 7)
