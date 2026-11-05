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

from typing import Annotated, Any, Protocol, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.chat.llm import get_llm
from app.models import EngineResult, Message, Source
from app.tools import get_tools
from app.vectordb.retriever import Retriever, get_retriever

PLACEHOLDER_ANSWER = "Engine not implemented yet — see app/chat/engine.py"

SYSTEM_PROMPT = (
    "You are a helpful assistant for the employees of a company. "
    "Answer clearly and concisely. If you do not know the answer, say so instead of guessing."
)


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


class GraphState(TypedDict):
    """What flows between the graph's nodes during one chat turn."""

    messages: Annotated[list[BaseMessage], add_messages]  # history + new question + replies
    context: str  # retrieved document text for the prompt (milestone 7)
    sources: list[Source]  # retrieved chunks shown under the answer (milestone 7)


_ROLE_TO_MESSAGE: dict[str, type[BaseMessage]] = {"user": HumanMessage, "assistant": AIMessage}


def to_langchain_messages(history: list[Message]) -> list[BaseMessage]:
    """Convert stored session messages to LangChain messages, oldest first.

    Stored `system` messages and empty messages are skipped: the engine adds its own
    system prompt, and providers reject empty turns.
    """
    return [
        _ROLE_TO_MESSAGE[m.role](content=m.content) for m in history if m.role in _ROLE_TO_MESSAGE and m.content.strip()
    ]


class LangGraphEngine:
    """Chat engine backed by a compiled LangGraph graph: START -> generate -> END."""

    def __init__(self, llm: BaseChatModel, retriever: Retriever | None, tools: list[BaseTool]) -> None:
        self.llm = llm
        self.retriever = retriever  # wired into the graph in milestone 7
        self.tools = tools  # bound to the LLM in milestone 8
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(GraphState)
        builder.add_node("generate", self._generate)
        builder.add_edge(START, "generate")
        builder.add_edge("generate", END)
        return builder.compile()

    def _generate(self, state: GraphState) -> dict[str, Any]:
        prompt = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = self.llm.invoke(prompt)
        return {"messages": [response]}

    def invoke(self, message: str, history: list[Message], session_id: str) -> EngineResult:
        messages = [*to_langchain_messages(history), HumanMessage(content=message)]
        initial: GraphState = {"messages": messages, "context": "", "sources": []}
        final_state = self.graph.invoke(initial, config={"metadata": {"session_id": session_id}})
        answer = final_state["messages"][-1].text
        return EngineResult(answer=answer, sources=final_state.get("sources", []))


def _default_retriever() -> Retriever | None:
    try:
        return get_retriever()
    except NotImplementedError:
        return None


def _default_tools() -> list[BaseTool]:
    try:
        return get_tools()
    except NotImplementedError:
        return []


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

    TODO(milestone 7): add the `retrieve` node and fill `sources`.
    TODO(milestone 8): bind tools and add the tool loop.
    """
    return LangGraphEngine(
        llm=llm if llm is not None else get_llm(),
        retriever=retriever if retriever is not None else _default_retriever(),
        tools=tools if tools is not None else _default_tools(),
    )
