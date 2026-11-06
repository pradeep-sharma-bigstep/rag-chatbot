"""MILESTONE 2 — Observability with LangSmith.

LangChain/LangGraph send traces to LangSmith automatically once tracing is on.
Your job is to switch it on safely from `.env` and make traces easy to find.

  LANGSMITH_TRACING=true          (the older name LANGCHAIN_TRACING_V2=true also counts)
  LANGSMITH_API_KEY=lsv2_...
  LANGSMITH_PROJECT=rag-chatbot-<your-name>
  LANGSMITH_ENDPOINT=...          (optional; the evaluator points this at a local fake server)

Rules the evaluator checks (against a local fake LangSmith server, no account needed):
  - `configure_tracing()` returns True only when tracing is switched on AND an API key is set,
    and False otherwise (it must never crash the app when keys are missing).
  - A chat turn produces a trace in the project named by LANGSMITH_PROJECT.
  - The trace carries `session_id` in its metadata (see app/chat/engine.py).
  - The trace contains an LLM run, and your retriever appears as a "retriever" run.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

from langsmith import utils as ls_utils

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "rag-chatbot"
FLAG_VARS = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")  # new name first, legacy second
KEY_VARS = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
PROJECT_VARS = ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT")
_TRUTHY = frozenset({"true", "1", "yes", "on"})


def _first_env(names: Iterable[str]) -> str:
    """The first non-blank value among `names`, or ""."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _tracing_requested() -> bool:
    return any(os.getenv(name, "").strip().lower() in _TRUTHY for name in FLAG_VARS)


def _set_flags(value: str) -> None:
    """Write `value` to every flag variable that is set.

    LangSmith only treats the exact string "true" as on, so "True" or "1" are normalised
    here; "false" switches tracing off for real when the key is missing.
    """
    for name in FLAG_VARS:
        if name in os.environ:
            os.environ[name] = value


def _clear_langsmith_env_cache() -> None:
    """langsmith caches env lookups; clear them so the values set here take effect."""
    for cached in (ls_utils.get_env_var, ls_utils.get_tracer_project):
        cache_clear = getattr(cached, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()


def _configure() -> bool:
    if not _tracing_requested():
        return False
    if not _first_env(KEY_VARS):
        logger.warning("LangSmith tracing requested but LANGSMITH_API_KEY is not set; tracing is off.")
        _set_flags("false")
        return False
    _set_flags("true")
    os.environ["LANGSMITH_PROJECT"] = _first_env(PROJECT_VARS) or DEFAULT_PROJECT
    logger.info("LangSmith tracing on (project=%s)", os.environ["LANGSMITH_PROJECT"])
    return True


def configure_tracing() -> bool:
    """Called once at app start-up (see app/main.py).

    OUT: True if LangSmith tracing is active, False if it is off or misconfigured.
         Never raises: observability must not take the app down.
    """
    try:
        return _configure()
    except Exception:
        logger.exception("Could not configure LangSmith tracing; continuing without it")
        return False
    finally:
        _clear_langsmith_env_cache()
