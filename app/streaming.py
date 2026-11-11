"""Streaming version of a chat turn (milestone 9): yields the answer piece by piece, then saves it."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from app.chat.engine import ChatEngine, Finished, Restart, StreamEvent, Token
from app.models import EngineResult, Message
from app.storage.json_store import SessionStore

logger = logging.getLogger(__name__)


def _engine_events(engine: ChatEngine, message: str, history: list[Message], session_id: str) -> Iterator[StreamEvent]:
    """The engine's own stream, or the whole answer as one piece for an engine that cannot stream."""
    stream = getattr(engine, "stream", None)
    if stream is not None:
        yield from stream(message=message, history=history, session_id=session_id)
        return
    result = engine.invoke(message=message, history=history, session_id=session_id)
    yield Token(result.answer)
    yield Finished(result)


def stream_chat_turn(
    store: SessionStore, engine: ChatEngine, session_id: str, content: str
) -> Iterator[Token | Restart | Message]:
    """Send one user message and yield Token/Restart events, then the saved assistant Message last.

    Raises SessionNotFoundError right away (before the first event), so callers can answer 404.
    As in `run_chat_turn`, an engine failure becomes the assistant message instead of an exception.
    If the caller stops reading early (the browser closed), nothing is saved for this turn.
    """
    history = store.get(session_id).messages
    return _turn(store, engine, session_id, content, history)


def _turn(
    store: SessionStore, engine: ChatEngine, session_id: str, content: str, history: list[Message]
) -> Iterator[Token | Restart | Message]:
    user_message = Message(role="user", content=content)
    result: EngineResult | None = None
    try:
        for event in _engine_events(engine, content, history, session_id):
            if isinstance(event, Finished):
                result = event.result
            else:
                yield event
    except Exception as exc:
        logger.exception("Engine failed for session %s", session_id)
        assistant = Message(role="assistant", content=f"Engine error: {type(exc).__name__}: {exc}")
    else:
        if result is None:
            assistant = Message(role="assistant", content="Engine error: no answer was produced")
        else:
            assistant = Message(role="assistant", content=result.answer, sources=result.sources)
    store.append_messages(session_id, user_message, assistant)
    yield assistant
