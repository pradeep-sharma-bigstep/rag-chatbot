"""Use-cases shared by the JSON API and the HTML pages. Complete — no TODOs here."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.chat.engine import ChatEngine
from app.config import Settings
from app.deps import Components
from app.ingest.pipeline import ingest_file
from app.models import Message
from app.storage.json_store import SessionStore

logger = logging.getLogger(__name__)
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def run_chat_turn(store: SessionStore, engine: ChatEngine, session_id: str, content: str) -> tuple[Message, Message]:
    """Send one user message through the engine and save both messages."""
    session = store.get(session_id)  # raises SessionNotFoundError before we call the engine
    user_message = Message(role="user", content=content)
    try:
        result = engine.invoke(message=content, history=session.messages, session_id=session_id)
        assistant_message = Message(role="assistant", content=result.answer, sources=result.sources)
    except Exception as exc:
        logger.exception("Engine failed for session %s", session_id)
        assistant_message = Message(role="assistant", content=f"Engine error: {type(exc).__name__}: {exc}")
    store.append_messages(session_id, user_message, assistant_message)
    return user_message, assistant_message


class UploadError(ValueError):
    pass


@dataclass
class UploadResult:
    filename: str
    indexed: bool
    chunks_indexed: int
    detail: str


def save_and_index_upload(settings: Settings, components: Components, filename: str, data: bytes) -> UploadResult:
    """Save an uploaded file into DOCUMENTS_DIR and index it if ingestion is available."""
    safe_name = _SAFE_NAME.sub("_", Path(filename or "").name).strip("._")
    suffix = Path(safe_name).suffix.lower()
    if not safe_name or suffix not in settings.allowed_upload_suffixes:
        allowed = ", ".join(settings.allowed_upload_suffixes)
        raise UploadError(f"Unsupported file type. Allowed: {allowed}")
    if len(data) > settings.max_upload_bytes:
        raise UploadError("File too large")
    if not data.strip():
        raise UploadError("File is empty")

    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    path = settings.documents_dir / safe_name
    path.write_bytes(data)

    if not components.can_index:
        missing = ", ".join(sorted(k for k in ("chunker", "embedder", "vector_store") if k in components.missing))
        return UploadResult(safe_name, False, 0, f"Saved. Indexing not available yet ({missing}).")
    assert components.chunker and components.embedder and components.store
    try:
        count = ingest_file(path, chunker=components.chunker, embedder=components.embedder, store=components.store)
    except NotImplementedError as exc:
        return UploadResult(safe_name, False, 0, f"Saved. Indexing not available yet ({exc}).")
    return UploadResult(safe_name, True, count, f"Saved and indexed {count} chunks.")
