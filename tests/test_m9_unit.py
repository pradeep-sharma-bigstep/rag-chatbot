"""Milestone 9 unit tests: streaming the answer (engine.stream, stream_chat_turn, the SSE endpoint, the page)."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool

from app.chat.engine import TOO_MANY_STEPS_ANSWER, Finished, Restart, Token, get_engine
from app.deps import Components, get_components
from app.models import EngineResult, Message
from app.storage.json_store import SessionNotFoundError, SessionStore
from app.streaming import stream_chat_turn
from app.tools import get_tools
from tests.fakes import FakeChatModel, StaticRetriever, make_chunk


class Streaming(BaseChatModel):
    """Plays back prepared AIMessages, streaming text word by word and tool calls as tool-call chunks."""

    script: Any  # an iterator of AIMessage

    @property
    def _llm_type(self) -> str:
        return "streaming-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Streaming:  # type: ignore[override]
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=next(self.script))])

    def _stream(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        message = next(self.script)
        for word in re.findall(r"\S+\s*", str(message.content)):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word))
        for index, call in enumerate(message.tool_calls):
            chunk = ToolCallChunk(name=call["name"], args=json.dumps(call["args"]), id=call["id"], index=index)
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[chunk]))


def streaming(*messages: AIMessage) -> Streaming:
    return Streaming(script=iter(messages))


def tool_call(name: str, args: dict[str, Any], call_id: str = "call1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def texts(events: list[Any]) -> list[str]:
    return [e.text for e in events if isinstance(e, Token)]


# ----------------------------------------------------------------------------- engine.stream


def test_stream_yields_the_answer_in_several_pieces_then_finished() -> None:
    engine = get_engine(llm=streaming(AIMessage(content="one two three four")), retriever=None, tools=[])
    events = list(engine.stream("hi", [], "s1"))  # type: ignore[attr-defined]
    assert len(texts(events)) > 1
    assert "".join(texts(events)) == "one two three four"
    assert isinstance(events[-1], Finished) and events[-1].result.answer == "one two three four"
    assert sum(isinstance(e, Finished) for e in events) == 1


def test_stream_answer_matches_invoke() -> None:
    llm = FakeChatModel(reply="same either way")
    engine = get_engine(llm=llm, retriever=None, tools=[])
    streamed = list(engine.stream("hi", [], "s"))[-1]  # type: ignore[attr-defined]
    assert isinstance(streamed, Finished)
    assert streamed.result == engine.invoke("hi", [], "s")


def test_a_model_that_cannot_stream_still_gives_the_text_once_without_duplicates() -> None:
    engine = get_engine(llm=FakeChatModel(reply="whole answer"), retriever=None, tools=[])
    events = list(engine.stream("hi", [], "s"))  # type: ignore[attr-defined]
    assert texts(events) == ["whole answer"]


def test_stream_returns_the_sources_and_sends_history_and_context_to_the_model() -> None:
    llm = FakeChatModel(reply="ok")
    retriever = StaticRetriever([make_chunk("Fridays are half days.", source="policy.md", position=3)])
    engine = get_engine(llm=llm, retriever=retriever, tools=[])
    history = [Message(role="user", content="earlier question"), Message(role="assistant", content="earlier answer")]
    final = list(engine.stream("when?", history, "s"))[-1]  # type: ignore[attr-defined]
    assert isinstance(final, Finished)
    assert [s.source for s in final.result.sources] == ["policy.md"]
    assert "earlier answer" in llm.prompt_text() and "Fridays are half days." in llm.prompt_text()


def test_stream_runs_tools_and_streams_only_the_final_answer() -> None:
    llm = streaming(tool_call("calculator", {"expression": "6 * 7"}), AIMessage(content="It is 42 today"))
    engine = get_engine(llm=llm, retriever=None, tools=get_tools())
    events = list(engine.stream("6*7?", [], "s"))  # type: ignore[attr-defined]
    assert "".join(texts(events)) == "It is 42 today"
    assert not any(isinstance(e, Restart) for e in events)
    assert events[-1].result.answer == "It is 42 today"  # type: ignore[union-attr]


def test_text_written_by_a_model_inside_a_tool_is_not_streamed_to_the_user() -> None:
    inner = Streaming(script=iter([AIMessage(content="secret inner words")]))

    @tool
    def summarise(text: str) -> str:
        """Summarise text with a helper model."""
        return str(inner.invoke(text).content).upper()

    llm = streaming(tool_call("summarise", {"text": "abc"}), AIMessage(content="Done"))
    engine = get_engine(llm=llm, retriever=None, tools=[summarise])
    events = list(engine.stream("go", [], "s"))  # type: ignore[attr-defined]
    assert "".join(texts(events)) == "Done"


def test_text_before_a_tool_call_is_cleared_with_restart() -> None:
    lead_in = AIMessage(
        content="Let me work that out",
        tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "c1"}],
    )
    llm = streaming(lead_in, AIMessage(content="The result is 2"))
    engine = get_engine(llm=llm, retriever=None, tools=get_tools())
    events = list(engine.stream("1+1?", [], "s"))  # type: ignore[attr-defined]
    assert "".join(texts(events[: events.index(Restart())])) == "Let me work that out"
    assert "".join(texts(events[events.index(Restart()) + 1 :])) == "The result is 2"
    assert events[-1].result.answer == "The result is 2"  # type: ignore[union-attr]


def test_a_model_that_never_stops_calling_tools_ends_with_the_friendly_answer() -> None:
    def forever() -> Iterator[AIMessage]:
        number = 0
        while True:
            number += 1
            yield tool_call("calculator", {"expression": "1+1"}, f"c{number}")

    engine = get_engine(llm=Streaming(script=forever()), retriever=None, tools=get_tools())
    events = list(engine.stream("loop", [], "s"))  # type: ignore[attr-defined]
    assert isinstance(events[-1], Finished) and events[-1].result.answer == TOO_MANY_STEPS_ANSWER
    assert sum(isinstance(e, Finished) for e in events) == 1


def test_stream_is_traced_with_the_session_id() -> None:
    from langchain_core.tracers.context import collect_runs

    engine = get_engine(llm=FakeChatModel(reply="x"), retriever=None, tools=[])
    with collect_runs() as cb:
        list(engine.stream("hi", [], "session-9"))  # type: ignore[attr-defined]
    roots = [r for r in cb.traced_runs if r.name == "chat_turn"]
    assert roots and roots[0].metadata["session_id"] == "session-9"


# ----------------------------------------------------------------------------- stream_chat_turn


class ExplodingStream:
    graph: Any = None

    def invoke(self, message: str, history: list[Message], session_id: str) -> EngineResult:
        raise AssertionError("unused")

    def stream(self, message: str, history: list[Message], session_id: str) -> Iterator[Any]:
        yield Token("par")
        raise RuntimeError("model fell over")


class InvokeOnly:
    graph: Any = None

    def invoke(self, message: str, history: list[Message], session_id: str) -> EngineResult:
        return EngineResult(answer=f"echo {message}", sources=[])


@pytest.fixture()
def store(tmp_path: Any) -> SessionStore:
    return SessionStore(tmp_path / "sessions")


def test_stream_chat_turn_yields_tokens_then_the_saved_message_and_stores_both(store: SessionStore) -> None:
    session = store.create()
    engine = get_engine(llm=streaming(AIMessage(content="a b c")), retriever=None, tools=[])
    events = list(stream_chat_turn(store, engine, session.id, "question"))
    last = events[-1]
    assert isinstance(last, Message) and last.role == "assistant" and last.content == "a b c"
    assert all(isinstance(e, Token) for e in events[:-1])
    saved = store.get(session.id).messages
    assert [(m.role, m.content) for m in saved] == [("user", "question"), ("assistant", "a b c")]


def test_missing_session_raises_before_any_event(store: SessionStore) -> None:
    with pytest.raises(SessionNotFoundError):
        stream_chat_turn(store, InvokeOnly(), "nope", "hi")  # raised by the call itself, not by reading it


def test_an_engine_that_cannot_stream_is_sent_as_one_piece(store: SessionStore) -> None:
    session = store.create()
    events = list(stream_chat_turn(store, InvokeOnly(), session.id, "hello"))
    assert [e.text for e in events if isinstance(e, Token)] == ["echo hello"]
    assert isinstance(events[-1], Message) and events[-1].content == "echo hello"


def test_a_failing_engine_becomes_the_saved_assistant_message(store: SessionStore) -> None:
    session = store.create()
    events = list(stream_chat_turn(store, ExplodingStream(), session.id, "hi"))
    final = events[-1]
    assert isinstance(final, Message) and final.content == "Engine error: RuntimeError: model fell over"
    assert [m.role for m in store.get(session.id).messages] == ["user", "assistant"]


def test_stopping_early_saves_nothing(store: SessionStore) -> None:
    session = store.create()
    engine = get_engine(llm=streaming(AIMessage(content="a b c d e")), retriever=None, tools=[])
    events = stream_chat_turn(store, engine, session.id, "hi")
    next(events)
    events.close()  # type: ignore[attr-defined]
    assert store.get(session.id).messages == []


def test_history_reaches_the_model_in_a_streamed_turn(store: SessionStore) -> None:
    session = store.create()
    store.append_messages(
        session.id, Message(role="user", content="my name is Asha"), Message(role="assistant", content="Hi")
    )
    llm = FakeChatModel(reply="Asha")
    list(stream_chat_turn(store, get_engine(llm=llm, retriever=None, tools=[]), session.id, "who am I?"))
    assert "my name is Asha" in llm.prompt_text()


# ----------------------------------------------------------------------------- HTTP / SSE


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.split("\n")
        assert lines[0].startswith("event: ") and lines[1].startswith("data: "), block
        events.append((lines[0][7:], json.loads(lines[1][6:])))
    return events


def use_engine(client: TestClient, engine: Any) -> None:
    components = Components(engine=engine)
    client.app.dependency_overrides[get_components] = lambda: components  # type: ignore[attr-defined]


def test_stream_endpoint_sends_token_events_then_done(client: TestClient) -> None:
    use_engine(client, get_engine(llm=streaming(AIMessage(content="hello there friend")), retriever=None, tools=[]))
    sid = client.post("/api/sessions").json()["id"]
    response = client.post(f"/api/sessions/{sid}/messages/stream", json={"content": "hi"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    events = parse_sse(response.text)
    kinds = [kind for kind, _ in events]
    assert kinds[-1] == "done" and kinds.count("done") == 1 and kinds.count("token") > 1
    assert "".join(data["text"] for kind, data in events if kind == "token") == "hello there friend"
    message = events[-1][1]["message"]
    assert message["role"] == "assistant" and message["content"] == "hello there friend"


def test_stream_endpoint_saves_the_turn_like_the_normal_endpoint(client: TestClient) -> None:
    use_engine(client, get_engine(llm=FakeChatModel(reply="saved reply"), retriever=None, tools=[]))
    sid = client.post("/api/sessions").json()["id"]
    client.post(f"/api/sessions/{sid}/messages/stream", json={"content": "hi"})
    saved = client.get(f"/api/sessions/{sid}").json()["messages"]
    assert [(m["role"], m["content"]) for m in saved] == [("user", "hi"), ("assistant", "saved reply")]


def test_done_event_carries_the_sources(client: TestClient) -> None:
    retriever = StaticRetriever([make_chunk("Fridays are half days.", source="policy.md")])
    use_engine(client, get_engine(llm=FakeChatModel(reply="ok"), retriever=retriever, tools=[]))
    sid = client.post("/api/sessions").json()["id"]
    events = parse_sse(client.post(f"/api/sessions/{sid}/messages/stream", json={"content": "when?"}).text)
    assert [s["source"] for s in events[-1][1]["message"]["sources"]] == ["policy.md"]


def test_newlines_and_quotes_in_the_answer_survive_the_event_format(client: TestClient) -> None:
    reply = 'line one\n\nline "two"\n- item'
    use_engine(client, get_engine(llm=FakeChatModel(reply=reply), retriever=None, tools=[]))
    sid = client.post("/api/sessions").json()["id"]
    events = parse_sse(client.post(f"/api/sessions/{sid}/messages/stream", json={"content": "hi"}).text)
    assert "".join(d["text"] for k, d in events if k == "token") == reply
    assert events[-1][1]["message"]["content"] == reply


def test_engine_failure_is_reported_in_done_with_status_200(client: TestClient) -> None:
    use_engine(client, ExplodingStream())
    sid = client.post("/api/sessions").json()["id"]
    response = client.post(f"/api/sessions/{sid}/messages/stream", json={"content": "hi"})
    assert response.status_code == 200
    assert parse_sse(response.text)[-1][1]["message"]["content"].startswith("Engine error: RuntimeError")


def test_unknown_session_is_404_and_bad_bodies_are_422(client: TestClient) -> None:
    use_engine(client, InvokeOnly())
    assert client.post("/api/sessions/nope/messages/stream", json={"content": "hi"}).status_code == 404
    sid = client.post("/api/sessions").json()["id"]
    assert client.post(f"/api/sessions/{sid}/messages/stream", json={"content": ""}).status_code == 422
    assert client.post(f"/api/sessions/{sid}/messages/stream", json={"content": "x" * 8001}).status_code == 422
    assert client.post(f"/api/sessions/{sid}/messages/stream", json={}).status_code == 422


def test_the_normal_endpoint_still_works(client: TestClient) -> None:
    use_engine(client, get_engine(llm=FakeChatModel(reply="plain"), retriever=None, tools=[]))
    sid = client.post("/api/sessions").json()["id"]
    turn = client.post(f"/api/sessions/{sid}/messages", json={"content": "hi"}).json()
    assert turn["assistant_message"]["content"] == "plain"


# ----------------------------------------------------------------------------- the page


def test_page_points_the_composer_at_the_stream_endpoint_and_loads_the_script(client: TestClient) -> None:
    sid = client.post("/api/sessions").json()["id"]
    html = client.get(f"/sessions/{sid}").text
    assert f'data-stream-url="/api/sessions/{sid}/messages/stream"' in html
    assert 'src="http://testserver/static/chat.js"' in html
    assert 'id="composer"' in html


def test_the_script_is_served_and_never_uses_innerhtml(client: TestClient) -> None:
    response = client.get("/static/chat.js")
    assert response.status_code == 200
    code = "\n".join(line.split("//")[0] for line in response.text.splitlines())  # ignore comments
    assert "innerHTML" not in code and "insertAdjacentHTML" not in code  # model text is untrusted


def test_the_html_form_for_messages_still_works_without_javascript(client: TestClient) -> None:
    use_engine(client, get_engine(llm=FakeChatModel(reply="html reply"), retriever=None, tools=[]))
    sid = client.post("/api/sessions").json()["id"]
    response = client.post(f"/ui/sessions/{sid}/messages", data={"content": "hi"})
    assert response.status_code == 200 and "html reply" in response.text
