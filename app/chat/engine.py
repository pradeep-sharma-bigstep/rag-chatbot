"""MILESTONES 1, 7, 8 — The chat engine (a LangGraph graph).

The HTTP layer (app/api/chat.py) calls:

    result = engine.invoke(message=..., history=[...], session_id=...)

and saves `result.answer` and `result.sources` into the session file.

Build it up in stages:
  M1  graph: START -> generate -> END. `generate` calls the LLM with the history
      plus the new user message.
  M7  add a `retrieve` node BEFORE `generate`: call `retriever.retrieve(message)`,
      put the chunk texts into the prompt, and return them as `sources`.
      (Retrieval must happen in the graph on every turn — not only when the LLM asks.)
  M8  bind `tools` to the LLM and add a tool node (langgraph.prebuilt.ToolNode +
      tools_condition) so the LLM can call a tool and then answer.

Rules the evaluator checks (with a scripted fake LLM, so no API key needed):
  - `get_engine(llm=..., retriever=..., tools=...)` uses the objects it is GIVEN.
  - `engine.graph` is the compiled LangGraph graph (`StateGraph(...).compile()`).
  - `invoke` returns EngineResult; `answer` is the LLM's final message text.
  - The previous history and the retrieved chunk text both reach the LLM.
  - When the LLM asks for a tool, the tool runs and the LLM is called again.
  - `sources` lists the retrieved chunks (source file + chunk id).
  - The graph is invoked with `config={"metadata": {"session_id": session_id}}`
    so the LangSmith trace can be filtered by session.
"""

from __future__ import annotations

from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from app.models import EngineResult, Message
from app.vectordb.retriever import Retriever

PLACEHOLDER_ANSWER = "Engine not implemented yet — see app/chat/engine.py"


class ChatEngine(Protocol):
    graph: Any  # the compiled LangGraph graph

    def invoke(self, message: str, history: list[Message], session_id: str) -> EngineResult:
        """IN: new user message, earlier messages of this session, session id.
        OUT: EngineResult(answer=..., sources=[...])."""
        ...


class PlaceholderEngine:
    """Used by the app until `get_engine` works. Do not edit — replace it via get_engine."""

    graph: Any = None

    def invoke(self, message: str, history: list[Message], session_id: str) -> EngineResult:
        return EngineResult(answer=PLACEHOLDER_ANSWER, sources=[])


def get_engine(
    llm: BaseChatModel | None = None,
    retriever: Retriever | None = None,
    tools: list[BaseTool] | None = None,
) -> ChatEngine:
    """Build the chat engine.

    IN:  llm       None -> app.chat.llm.get_llm()
         retriever None -> app.vectordb.retriever.get_retriever() (if that still raises
                           NotImplementedError, run without retrieval)
         tools     None -> app.tools.get_tools() (same fallback: no tools)
    OUT: an object with `.graph` and `.invoke(message, history, session_id)`.

    TODO(milestone 1): define the graph state (TypedDict with `messages`, `context`, `sources`),
                       the `generate` node, compile the graph, wrap it in a class.
    TODO(milestone 7): add the `retrieve` node and fill `sources`.
    TODO(milestone 8): bind tools and add the tool loop.
    """
    raise NotImplementedError("TODO(milestone 1): implement get_engine in app/chat/engine.py")
