"""JSON API for chatting: POST /api/sessions/{id}/messages."""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.chat.engine import Restart, Token
from app.deps import Components, get_components, get_store
from app.models import Message
from app.services import run_chat_turn
from app.storage.json_store import SessionNotFoundError, SessionStore
from app.streaming import stream_chat_turn

router = APIRouter(prefix="/api/sessions", tags=["chat"])


class SendMessage(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


class ChatTurn(BaseModel):
    user_message: Message
    assistant_message: Message


@router.post("/{session_id}/messages", response_model=ChatTurn)
def send_message(
    session_id: str,
    body: SendMessage,
    store: SessionStore = Depends(get_store),
    components: Components = Depends(get_components),
) -> ChatTurn:
    try:
        user_message, assistant_message = run_chat_turn(store, components.engine, session_id, body.content)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    return ChatTurn(user_message=user_message, assistant_message=assistant_message)


def _sse(event: str, data: dict[str, object]) -> str:
    """One Server-Sent Event. JSON keeps newlines in the text from breaking the format."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _sse_stream(events: Iterator[Token | Restart | Message]) -> Iterator[str]:
    for event in events:
        if isinstance(event, Token):
            yield _sse("token", {"text": event.text})
        elif isinstance(event, Restart):
            yield _sse("restart", {})
        else:
            yield _sse("done", {"message": event.model_dump(mode="json")})


@router.post("/{session_id}/messages/stream")
def stream_message(
    session_id: str,
    body: SendMessage,
    store: SessionStore = Depends(get_store),
    components: Components = Depends(get_components),
) -> StreamingResponse:
    """Same as POST .../messages, but the answer arrives as Server-Sent Events.

    `token` {text} for each piece, `restart` to clear text that preceded a tool call, and a final
    `done` {message} holding the saved assistant message (with sources).
    """
    try:
        events = stream_chat_turn(store, components.engine, session_id, body.content)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(_sse_stream(events), media_type="text/event-stream", headers=headers)
