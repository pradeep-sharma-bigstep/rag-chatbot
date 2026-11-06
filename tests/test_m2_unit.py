"""Milestone 2 unit tests — LangSmith tracing switch and what a chat turn sends.

No LangSmith account needed: the last test points LANGSMITH_ENDPOINT at a tiny local
HTTP server that records what LangChain uploads.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import langsmith.run_trees as ls_run_trees
import pytest
from langchain_core.tracers.context import collect_runs
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import utils as ls_utils

from app import observability
from app.chat.engine import TRACE_RUN_NAME, get_engine
from app.observability import DEFAULT_PROJECT, configure_tracing
from tests.fakes import FakeChatModel

ENV_VARS = (
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGCHAIN_PROJECT",
    "LANGSMITH_ENDPOINT",
    "LANGCHAIN_ENDPOINT",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Start every test with no LangSmith variables and restore them exactly afterwards.

    setenv-then-delenv makes monkeypatch remember the original state even for variables
    that were absent, so anything configure_tracing() writes is undone after the test.
    """
    for name in ENV_VARS:
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    yield
    monkeypatch.undo()
    ls_utils.get_env_var.cache_clear()  # don't leave cached "tracing on" for later tests
    ls_utils.get_tracer_project.cache_clear()


# --------------------------------------------------------------------------- the switch


@pytest.mark.parametrize("flag", ["true", "TRUE", " True ", "1", "yes", "on"])
def test_truthy_flags_enable_and_are_normalised(monkeypatch: pytest.MonkeyPatch, flag: str) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", flag)
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    assert configure_tracing() is True
    assert os.environ["LANGSMITH_TRACING"] == "true", "LangSmith itself only accepts lowercase 'true'"
    assert ls_utils.tracing_is_enabled() is True


@pytest.mark.parametrize("flag", ["false", "0", "", "no", "off"])
def test_falsy_flags_stay_off_even_with_key(monkeypatch: pytest.MonkeyPatch, flag: str) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", flag)
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    assert configure_tracing() is False


def test_legacy_variable_names_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test")
    assert configure_tracing() is True


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_or_blank_key_warns_and_really_switches_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, key: str | None
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    if key is not None:
        monkeypatch.setenv("LANGSMITH_API_KEY", key)
    with caplog.at_level(logging.WARNING, logger="app.observability"):
        assert configure_tracing() is False
    assert "LANGSMITH_API_KEY" in caplog.text
    # Not just reported as off: LangChain must not try (and fail) to upload every run.
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert ls_utils.tracing_is_enabled() is False


def test_default_project_when_none_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    assert configure_tracing() is True
    assert os.environ["LANGSMITH_PROJECT"] == DEFAULT_PROJECT
    assert ls_utils.get_tracer_project() == DEFAULT_PROJECT


@pytest.mark.parametrize("var", ["LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"])
def test_given_project_is_kept(monkeypatch: pytest.MonkeyPatch, var: str) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.setenv(var, "rag-chatbot-priya")
    assert configure_tracing() is True
    assert ls_utils.get_tracer_project() == "rag-chatbot-priya"


def test_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> bool:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(observability, "_configure", boom)
    assert configure_tracing() is False


# --------------------------------------------------------------------------- what gets traced


def test_chat_turn_runs_carry_session_id() -> None:
    """Captured in memory: a named chat_turn run and an LLM run, both tagged with session_id."""
    engine = get_engine(llm=FakeChatModel(), retriever=None, tools=[])
    with collect_runs() as collector:
        engine.invoke("hi", [], "session-42")

    runs = {run.name: run for run in collector.traced_runs}
    assert TRACE_RUN_NAME in runs, f"expected a {TRACE_RUN_NAME!r} run, got {sorted(runs)}"
    assert runs[TRACE_RUN_NAME].run_type == "chain"
    llm_runs = [run for run in collector.traced_runs if run.run_type == "llm"]
    assert llm_runs, "the LLM call must be traced"
    for run in (runs[TRACE_RUN_NAME], *llm_runs):
        assert run.extra["metadata"]["session_id"] == "session-42"


class _RecordingHandler(BaseHTTPRequestHandler):
    bodies: list[bytes] = []

    def _record(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        self.bodies.append(self.rfile.read(length) if length else b"")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    do_GET = do_POST = do_PATCH = _record

    def log_message(self, *args: object) -> None:  # keep test output quiet
        pass


@pytest.fixture()
def fake_langsmith() -> Iterator[str]:
    _RecordingHandler.bodies = []
    server = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_chat_turn_is_uploaded_to_langsmith(monkeypatch: pytest.MonkeyPatch, fake_langsmith: str) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_fake")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", fake_langsmith)
    monkeypatch.setenv("LANGSMITH_PROJECT", "m2-unit-test-project")
    monkeypatch.setattr(ls_run_trees, "_CLIENT", None)  # fresh client that uses the fake endpoint
    assert configure_tracing() is True

    get_engine(llm=FakeChatModel(), retriever=None, tools=[]).invoke("hello", [], "session-xyz")
    wait_for_all_tracers()  # flush the background uploader

    uploaded = b"".join(_RecordingHandler.bodies)
    assert b"session-xyz" in uploaded, "trace must carry session_id metadata"
    assert b"m2-unit-test-project" in uploaded, "trace must go to LANGSMITH_PROJECT"
    assert b'"run_type":"llm"' in uploaded.replace(b" ", b""), "trace must contain an LLM run"
