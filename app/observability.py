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


def configure_tracing() -> bool:
    """Called once at app start-up (see app/main.py).

    OUT: True if LangSmith tracing is active, False if it is off or misconfigured.

    TODO(milestone 2):
      - Read LANGSMITH_TRACING (or LANGCHAIN_TRACING_V2) and LANGSMITH_API_KEY.
      - If tracing is wanted but the key is missing, log a warning and return False.
      - Set a default LANGSMITH_PROJECT if none is given.
    """
    return False
