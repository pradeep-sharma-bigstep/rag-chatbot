"""Milestone 1 unit tests — get_llm and the LangGraph engine. No API key or network needed."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.chat import engine as engine_module
from app.chat.engine import SYSTEM_PROMPT, LangGraphEngine, get_engine, to_langchain_messages
from app.chat.llm import get_llm
from app.config import get_settings
from app.deps import Components, get_components
from app.models import Message
from tests.fakes import FakeChatModel

# --------------------------------------------------------------------------- get_llm


@pytest.fixture()
def fresh_settings() -> Iterator[None]:
    """get_settings() is cached; clear it so monkeypatched env vars are picked up."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_get_llm_unknown_provider_raises(monkeypatch: pytest.MonkeyPatch, fresh_settings: None) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "foo")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_llm()


def test_get_llm_empty_model_raises(monkeypatch: pytest.MonkeyPatch, fresh_settings: None) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "")
    with pytest.raises(ValueError, match="LLM_MODEL"):
        get_llm()


def test_get_llm_groq_reads_model_from_env(monkeypatch: pytest.MonkeyPatch, fresh_settings: None) -> None:
    monkeypatch.setenv("LLM_PROVIDER", " Groq ")  # case- and whitespace-insensitive
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_real")
    llm = get_llm()  # building the client makes no network call
    assert isinstance(llm, ChatGroq)
    assert llm.model_name == "llama-3.3-70b-versatile"
    # ChatGroq maps temperature=0 to 1e-8 because Groq's API needs a value above zero.
    assert llm.temperature == pytest.approx(0, abs=1e-6), "answers should be repeatable"


def test_get_llm_groq_key_comes_from_env(monkeypatch: pytest.MonkeyPatch, fresh_settings: None) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_from_environment")
    llm = get_llm()
    assert isinstance(llm, ChatGroq)
    assert llm.groq_api_key is not None
    assert llm.groq_api_key.get_secret_value() == "gsk_from_environment"


# --------------------------------------------------------------------------- history conversion


def test_history_roles_are_mapped_in_order() -> None:
    history = [
        Message(role="user", content="Q1"),
        Message(role="assistant", content="A1"),
        Message(role="system", content="stored system note"),  # skipped
        Message(role="assistant", content="   "),  # empty, skipped
        Message(role="user", content="Q2"),
    ]
    converted = to_langchain_messages(history)
    assert [type(m) for m in converted] == [HumanMessage, AIMessage, HumanMessage]
    assert [m.content for m in converted] == ["Q1", "A1", "Q2"]


# --------------------------------------------------------------------------- engine


def test_prompt_is_system_then_history_then_question() -> None:
    llm = FakeChatModel()
    history = [Message(role="user", content="My name is Priya"), Message(role="assistant", content="Hi Priya")]
    get_engine(llm=llm, retriever=None, tools=[]).invoke("What is my name?", history, "s1")

    prompt = llm.calls[-1]
    assert isinstance(prompt[0], SystemMessage) and prompt[0].content == SYSTEM_PROMPT
    assert [type(m) for m in prompt[1:]] == [HumanMessage, AIMessage, HumanMessage]
    assert prompt[-1].content == "What is my name?"


def test_empty_history_returns_answer_and_no_sources() -> None:
    result = get_engine(llm=FakeChatModel(reply="hello"), retriever=None, tools=[]).invoke("hi", [], "s1")
    assert result.answer == "hello"
    assert result.sources == []


def test_answer_is_text_only_when_model_returns_content_blocks() -> None:
    """Some models return reasoning + text blocks; only the text must reach the user."""
    reply = AIMessage(
        content=[
            {"type": "thinking", "thinking": "secret reasoning", "signature": "x"},
            {"type": "text", "text": "Visible answer"},
        ]
    )
    llm = GenericFakeChatModel(messages=iter([reply]))
    result = get_engine(llm=llm, retriever=None, tools=[]).invoke("hi", [], "s1")
    assert result.answer == "Visible answer"


def test_turns_do_not_leak_between_invocations() -> None:
    """The engine is stateless: memory comes only from the history it is given."""
    llm = FakeChatModel()
    engine = get_engine(llm=llm, retriever=None, tools=[])
    engine.invoke("first secret question", [], "s1")
    engine.invoke("second question", [], "s2")
    assert "first secret question" not in "\n".join(str(m.content) for m in llm.calls[-1])


def test_graph_is_invoked_with_session_id_metadata() -> None:
    engine = get_engine(llm=FakeChatModel(), retriever=None, tools=[])
    assert isinstance(engine, LangGraphEngine)
    seen: list[Any] = []
    real_graph = engine.graph

    class SpyGraph:
        def invoke(self, state: Any, config: Any = None) -> Any:
            seen.append(config)
            return real_graph.invoke(state, config=config)

    engine.graph = SpyGraph()
    engine.invoke("hi", [], "session-42")
    assert seen[0]["metadata"]["session_id"] == "session-42"


def test_get_engine_defaults_when_arguments_are_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeChatModel(reply="from default llm")

    def not_built() -> Any:
        raise NotImplementedError

    monkeypatch.setattr(engine_module, "get_llm", lambda: fake)
    monkeypatch.setattr(engine_module, "get_retriever", not_built)
    monkeypatch.setattr(engine_module, "get_tools", not_built)

    engine = get_engine()
    assert isinstance(engine, LangGraphEngine)
    assert engine.llm is fake and engine.retriever is None and engine.tools == []
    assert engine.invoke("hi", [], "s1").answer == "from default llm"


# --------------------------------------------------------------------------- through the HTTP API


def test_api_chat_remembers_earlier_messages(client: TestClient) -> None:
    llm = FakeChatModel(reply="noted")
    components = Components(engine=get_engine(llm=llm, retriever=None, tools=[]))
    client.app.dependency_overrides[get_components] = lambda: components  # type: ignore[attr-defined]

    sid = client.post("/api/sessions").json()["id"]
    first = client.post(f"/api/sessions/{sid}/messages", json={"content": "My name is Priya"}).json()
    assert first["assistant_message"]["content"] == "noted"

    client.post(f"/api/sessions/{sid}/messages", json={"content": "What is my name?"})
    last_prompt = [str(m.content) for m in llm.calls[-1]]
    assert last_prompt[-3:] == ["My name is Priya", "noted", "What is my name?"]
    assert len(client.get(f"/api/sessions/{sid}").json()["messages"]) == 4
