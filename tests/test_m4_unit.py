"""Milestone 4 unit tests — the embedder wrapper and the real local model.

The first group uses a fake model, so it needs no download. The second group loads the real
fastembed model (about 70 MB, downloaded once into ~/.cache/fastembed) and is skipped when the
model cannot be loaded, for example offline on a fresh machine.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from app.config import get_settings
from app.vectordb import embeddings
from app.vectordb.embeddings import BATCH_SIZE, DEFAULT_MODEL, Embedder, FastEmbedEmbedder, get_embedder

# --------------------------------------------------------------------------- fake model


class FakeFastEmbed:
    """Stands in for fastembed.TextEmbedding: records calls, returns small numpy vectors."""

    embedding_size = 4

    def __init__(self) -> None:
        self.passage_calls: list[tuple[list[str], int]] = []
        self.query_calls: list[str] = []

    @staticmethod
    def _vector(text: str, offset: float = 0.0) -> Any:
        return np.array([len(text), text.count("a"), text.count("e"), offset], dtype=np.float32)

    def passage_embed(self, texts: list[str], batch_size: int = 256) -> Iterator[Any]:
        self.passage_calls.append((list(texts), batch_size))
        return (self._vector(t) for t in texts)

    def query_embed(self, query: str) -> Iterator[Any]:
        self.query_calls.append(query)
        return iter([self._vector(query, offset=1.0)])


@pytest.fixture()
def fake() -> FakeFastEmbed:
    return FakeFastEmbed()


@pytest.fixture()
def embedder(fake: FakeFastEmbed) -> FastEmbedEmbedder:
    return FastEmbedEmbedder("fake-model", model=fake)


def test_dimension_comes_from_the_model(embedder: FastEmbedEmbedder) -> None:
    assert embedder.dimension == 4


def test_embed_documents_keeps_order_and_returns_plain_lists(embedder: FastEmbedEmbedder) -> None:
    vectors = embedder.embed_documents(["aa", "bbbb", "c"])
    assert [v[0] for v in vectors] == [2.0, 4.0, 1.0], "one vector per text, in the same order"
    assert all(isinstance(v, list) and all(isinstance(x, float) for x in v) for v in vectors)


def test_embed_documents_empty_list_never_calls_the_model(embedder: FastEmbedEmbedder, fake: FakeFastEmbed) -> None:
    assert embedder.embed_documents([]) == []
    assert fake.passage_calls == []


def test_embed_documents_sends_one_batched_call_not_one_per_text(
    embedder: FastEmbedEmbedder, fake: FakeFastEmbed
) -> None:
    embedder.embed_documents([f"text {i}" for i in range(100)])
    assert len(fake.passage_calls) == 1
    texts, batch_size = fake.passage_calls[0]
    assert len(texts) == 100 and batch_size == BATCH_SIZE


def test_documents_and_queries_use_their_own_model_methods(embedder: FastEmbedEmbedder, fake: FakeFastEmbed) -> None:
    doc_vector = embedder.embed_documents(["same text"])[0]
    query_vector = embedder.embed_query("same text")
    assert fake.query_calls == ["same text"]
    assert len(query_vector) == embedder.dimension
    assert query_vector != doc_vector, "the fake gives queries a different last value, proving query_embed was used"


def test_embed_query_returns_one_flat_vector(embedder: FastEmbedEmbedder) -> None:
    vector = embedder.embed_query("hello")
    assert isinstance(vector, list) and len(vector) == 4 and isinstance(vector[0], float)


def test_embedder_satisfies_the_protocol(embedder: FastEmbedEmbedder) -> None:
    typed: Embedder = embedder
    assert typed.dimension == 4


# --------------------------------------------------------------------------- get_embedder / settings


@pytest.fixture()
def fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_get_embedder_uses_model_from_env(
    monkeypatch: pytest.MonkeyPatch, fresh_settings: None, fake: FakeFastEmbed
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(embeddings, "_load_model", lambda name: seen.append(name) or fake)
    monkeypatch.setenv("EMBEDDING_MODEL", " some/other-model ")
    embedder = get_embedder()
    assert seen == ["some/other-model"] and isinstance(embedder, FastEmbedEmbedder)


def test_get_embedder_falls_back_to_default_model_when_empty(
    monkeypatch: pytest.MonkeyPatch, fresh_settings: None, fake: FakeFastEmbed
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(embeddings, "_load_model", lambda name: seen.append(name) or fake)
    monkeypatch.setenv("EMBEDDING_MODEL", "")
    get_embedder()
    assert seen == [DEFAULT_MODEL]


def test_unknown_model_raises_a_clear_value_error(monkeypatch: pytest.MonkeyPatch, fresh_settings: None) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "no/such-model")
    embeddings._load_model.cache_clear()
    with pytest.raises(ValueError, match="Unsupported EMBEDDING_MODEL 'no/such-model'"):
        get_embedder()


def test_cache_dir_is_not_a_temp_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)
    assert embeddings._cache_dir().endswith(".cache/fastembed")
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", "/somewhere/else")
    assert embeddings._cache_dir() == "/somewhere/else"


# --------------------------------------------------------------------------- the real model


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@pytest.fixture(scope="module")
def real() -> Embedder:
    get_settings.cache_clear()
    try:
        return get_embedder()
    except Exception as exc:  # no network on a fresh machine, corrupted cache ...
        pytest.skip(f"real embedding model not available: {exc}")


def test_real_model_vectors_have_the_declared_size(real: Embedder) -> None:
    vectors = real.embed_documents(["one", "two", "three"])
    assert real.dimension == 384
    assert len(vectors) == 3 and all(len(v) == real.dimension for v in vectors)


def test_real_model_is_deterministic(real: Embedder) -> None:
    first = real.embed_documents(["How many leave days do I get?"])[0]
    again = real.embed_documents(["How many leave days do I get?"])[0]
    assert cosine(first, again) == pytest.approx(1.0, abs=1e-6)


def test_real_model_batching_does_not_change_vectors(real: Embedder) -> None:
    """A text must embed the same alone or inside a big batch (padding must not leak in)."""
    texts = ["short", "a much longer sentence " * 30, "mid length text here"]
    together = real.embed_documents(texts * 20)  # more than one batch
    alone = real.embed_documents([texts[1]])[0]
    assert cosine(together[1], alone) == pytest.approx(1.0, abs=1e-5)


def test_real_model_query_matches_the_same_document_text(real: Embedder) -> None:
    text = "How many days of annual leave do I get?"
    assert cosine(real.embed_query(text), real.embed_documents([text])[0]) > 0.95


def test_real_model_related_text_is_closer_than_unrelated(real: Embedder) -> None:
    query = real.embed_query("How many days of annual leave do I get?")
    related = cosine(query, real.embed_query("vacation days allowance"))
    unrelated = cosine(query, real.embed_query("object storage encryption"))
    assert related > unrelated + 0.1, "should be clearly closer, not just barely"


def test_real_model_finds_the_right_sample_document_chunk(real: Embedder) -> None:
    """End-to-end sanity for M7: a question lands nearest the chunk that answers it."""
    chunks = [
        "Full-time employees receive 24 days of paid annual leave per calendar year.",
        "Meals while travelling are reimbursed up to 50 USD per day with a receipt.",
        "Objects are encrypted at rest with AES-256 and in transit with TLS.",
        "On-call engineers receive a stipend of 200 USD per week of on-call duty.",
    ]
    vectors = real.embed_documents(chunks)
    for question, expected in [
        ("How many vacation days do I get each year?", 0),
        ("What is the daily limit for food when I travel?", 1),
        ("Is my stored data encrypted?", 2),
        ("How much extra pay for being on call?", 3),
    ]:
        query = real.embed_query(question)
        best = max(range(len(chunks)), key=lambda i: cosine(query, vectors[i]))
        assert best == expected, question


def test_real_model_handles_empty_text_and_very_long_text(real: Embedder) -> None:
    empty, huge = real.embed_documents(["", "word " * 5000])
    assert len(empty) == len(huge) == real.dimension
    assert not any(math.isnan(x) for x in empty + huge)
