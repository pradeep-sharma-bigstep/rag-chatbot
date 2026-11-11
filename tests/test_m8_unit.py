"""Milestone 8 unit tests — the calculator, the clock, and the tool loop in the engine.

No API key needed: scripted fake models decide which tools get called.
"""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.tracers.context import collect_runs

from app.chat import engine as engine_module
from app.chat.engine import (
    MAX_TOOL_ROUNDS,
    RECURSION_LIMIT,
    TOO_MANY_STEPS_ANSWER,
    LangGraphEngine,
    get_engine,
)
from app.deps import Components, get_components
from app.models import Message
from app.tools import clock as clock_module
from app.tools import get_tools
from app.tools.calculator import MAX_EXPRESSION_LENGTH, calculate, calculator
from app.tools.clock import current_datetime, current_time
from tests.fakes import TOOL_TRIGGER, FakeChatModel, StaticRetriever, make_chunk

# =========================================================================== calculator: correct results


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("  2+2  ", "4"),
        ("10 - 3 - 2", "5"),
        ("2 ** 3 ** 2", "512"),  # right-associative
        ("-3 ** 2", "-9"),  # like Python: -(3 ** 2)
        ("(-3) ** 2", "9"),
        ("--5", "5"),
        ("+7", "7"),
        ("7 // 2", "3"),
        ("-7 // 2", "-4"),
        ("7 % 3", "1"),
        ("10 / 4", "2.5"),
        ("6 / 3", "2"),
        ("0.1 + 0.2", "0.3"),
        ("1 / 3", "0.333333333333"),
        ("2 ** -1", "0.5"),
        ("1e3", "1000"),
        ("1847 * 392 + 15", "724039"),
        ("(12.5 + 3) * 4", "62"),
        ("0.15 * 240", "36"),
        ("9 ** 999 // 9 ** 998", "9"),
        ("1" + "0" * 50, "1" + "0" * 50),
    ],
)
def test_calculator_gives_exact_results(expression: str, expected: str) -> None:
    assert calculate(expression) == expected
    assert calculator.invoke({"expression": expression}) == expected


# =========================================================================== calculator: security


INJECTION = [
    "__import__('os').getcwd()",
    "__import__('os').system('echo pwned')",
    "open('/etc/passwd').read()",
    "exec('1')",
    "eval('1+1')",
    "().__class__.__bases__[0].__subclasses__()",
    "(1).__class__",
    "[x for x in range(3)]",
    "{1: 2}",
    "{1, 2}",
    "[1, 2][0]",
    "(1, 2)",
    "lambda: 1",
    "1 if True else 2",
    "1 < 2",
    "1 == 1",
    "not 1",
    "1 and 2",
    "'abc'",
    "'a' * 5",
    "b'x'",
    "f'{1}'",
    "True",
    "None",
    "1j",
    "x",
    "x = 1",
    "a.b",
    "print(1)",
    "abs(-1)",
    "import os",
    "1; 2",
    "1\n+ 2",
    "2 ^ 3",
    "~1",
    "5 & 3",
    "5 | 3",
    "1 << 100",
    "1,000",
    "...",
    "*[1]",
    "await x",
    "$1",
    "1 @ 2",
]


@pytest.mark.parametrize("expression", INJECTION)
def test_anything_that_is_not_plain_arithmetic_is_rejected(expression: str) -> None:
    assert calculate(expression).startswith("Error"), expression
    assert str(calculator.invoke({"expression": expression})).startswith("Error"), expression


def test_injection_has_no_side_effects(tmp_path: Path) -> None:
    marker = tmp_path / "pwned"
    for payload in (
        f"__import__('os').system('touch {marker}')",
        f"open('{marker}', 'w').write('x')",
        f"exec(\"open('{marker}', 'w').write('x')\")",
    ):
        assert calculate(payload).startswith("Error")
    assert not marker.exists(), "no code from the expression may ever run"


def test_the_module_never_uses_eval_or_exec() -> None:
    # `app.tools.calculator` is the tool object (it shadows the submodule), so load the module by name
    module = importlib.import_module("app.tools.calculator")
    source = Path(module.__file__ or "").read_text(encoding="utf-8")
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("#", '"')))
    assert "eval(" not in code.replace("ast.literal_eval", "") and "exec(" not in code


@pytest.mark.parametrize(
    "expression",
    [
        "9**9**9",
        "10**1001",
        "2**100000",
        "9 ** 99999999999",
        "(9**999)**9",
        "(10**100)**100",
        " * ".join(["2**999"] * 20),  # each factor is fine, the product is not
        "1e308 * 10",
        "10.0 ** 400",
        "1e400",
    ],
)
def test_resource_bombs_are_refused_quickly(expression: str) -> None:
    started = time.perf_counter()
    assert calculate(expression).startswith("Error"), expression
    assert time.perf_counter() - started < 1.0, "must be refused quickly, not computed"


def test_big_but_reasonable_results_still_work() -> None:
    assert calculate("2 ** 1000").startswith("1071508607186267")
    assert calculate("9 ** 999").isdigit()


@pytest.mark.parametrize("expression", ["1 / 0", "1 // 0", "1 % 0", "0 ** -1", "5 / (3 - 3)", "1.5 / 0.0"])
def test_division_by_zero_is_an_error_message(expression: str) -> None:
    assert calculate(expression) == "Error: division by zero"


def test_results_that_are_not_real_numbers_are_errors() -> None:
    assert calculate("(-8) ** 0.5") == "Error: result is not a real number"


@pytest.mark.parametrize("expression", ["", "   ", "\n", "(", ")", "2 +", "* 3", "2 3", "()"])
def test_malformed_or_empty_input_is_an_error(expression: str) -> None:
    assert calculate(expression).startswith("Error")


def test_overlong_input_is_rejected_before_parsing() -> None:
    assert "too long" in calculate("1+" * MAX_EXPRESSION_LENGTH + "1")
    assert calculate("1+" * 50 + "1") == "51", "just under the limit is fine"


def test_deeply_nested_input_never_crashes() -> None:
    for expression in ("(" * 150 + "1" + ")" * 150, "-" * 190 + "1", "+".join(["(1"] * 90) + ")" * 90):
        assert isinstance(calculate(expression), str)


@pytest.mark.parametrize("value", [123, None, 1.5, ["2+2"], {"a": 1}, True])
def test_wrong_argument_types_come_back_as_error_strings_not_exceptions(value: Any) -> None:
    assert str(calculator.invoke({"expression": value})).startswith("Error")
    assert calculate(value).startswith("Error")  # type: ignore[arg-type]


def test_bytes_are_not_evaluated_by_the_function_itself() -> None:
    assert calculate(b"2+2").startswith("Error")  # type: ignore[arg-type]


def test_missing_or_extra_arguments_come_back_as_error_strings() -> None:
    assert str(calculator.invoke({})).startswith("Error")
    assert str(calculator.invoke({"wrong_name": "2+2"})).startswith("Error")
    assert calculator.invoke({"expression": "2+2", "extra": 1}) == "4", "unknown extra arguments are ignored"


def test_calculator_describes_itself_for_the_model() -> None:
    assert calculator.name == "calculator" and len(calculator.description) > 100
    assert "**" in calculator.description and "Error" in calculator.description
    assert set(calculator.args) == {"expression"} and calculator.args["expression"]["type"] == "string"


# =========================================================================== clock

FROZEN_UTC = datetime(2026, 11, 12, 4, 30, 15, 987654, tzinfo=UTC)
LOCAL_IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture()
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_now(zone: ZoneInfo | None) -> datetime:
        return FROZEN_UTC.astimezone(zone) if zone is not None else FROZEN_UTC.astimezone(LOCAL_IST)

    monkeypatch.setattr(clock_module, "_now", fake_now)


@pytest.mark.usefixtures("frozen_clock")
@pytest.mark.parametrize(
    "zone,expected",
    [
        (None, "2026-11-12T10:00:15+05:30"),
        ("", "2026-11-12T10:00:15+05:30"),
        ("  ", "2026-11-12T10:00:15+05:30"),
        ("local", "2026-11-12T10:00:15+05:30"),
        ("None", "2026-11-12T10:00:15+05:30"),
        ("UTC", "2026-11-12T04:30:15+00:00"),
        ("Asia/Tokyo", "2026-11-12T13:30:15+09:00"),
        ("America/New_York", "2026-11-11T23:30:15-05:00"),  # still the 11th there, and after the November DST end
        ("Europe/London", "2026-11-12T04:30:15+00:00"),
        (" Asia/Tokyo ", "2026-11-12T13:30:15+09:00"),
    ],
)
def test_current_time_in_a_timezone(zone: str | None, expected: str) -> None:
    assert current_time(zone) == expected


@pytest.mark.usefixtures("frozen_clock")
def test_output_has_no_microseconds_and_parses_back() -> None:
    value = current_time("UTC")
    assert "." not in value
    assert datetime.fromisoformat(value) == FROZEN_UTC.replace(microsecond=0)


@pytest.mark.parametrize(
    "zone", ["Nowhere/City", "Mars/Olympus", "../etc/passwd", "UTC+5", "x" * 300, "/abs/path", "a\x00b"]
)
def test_unknown_or_malformed_timezones_are_error_strings(zone: str) -> None:
    result = current_time(zone)
    assert result.startswith("Error: unknown timezone")


@pytest.mark.parametrize("value", [5, ["UTC"], {"tz": 1}, True])
def test_wrong_timezone_types_come_back_as_error_strings(value: Any) -> None:
    assert str(current_datetime.invoke({"timezone": value})).startswith("Error")


def test_the_real_clock_is_close_to_now() -> None:
    for arguments in ({}, {"timezone": "UTC"}, {"timezone": None}):
        parsed = datetime.fromisoformat(str(current_datetime.invoke(arguments)))
        assert parsed.tzinfo is not None, "always carries a UTC offset"
        assert abs(parsed - datetime.now(UTC)) < timedelta(seconds=5)


def test_clock_describes_itself_for_the_model() -> None:
    assert current_datetime.name == "current_datetime" and "ISO 8601" in current_datetime.description
    assert "timezone" in current_datetime.args


def test_zoneinfo_database_is_available() -> None:
    try:
        ZoneInfo("Asia/Kolkata")
    except ZoneInfoNotFoundError:  # pragma: no cover
        pytest.fail("this machine has no timezone database; `uv add tzdata` would fix the clock tool")


# =========================================================================== registry


def test_get_tools_registry() -> None:
    tools = get_tools()
    assert [t.name for t in tools] == ["calculator", "current_datetime"]
    assert all(isinstance(t, BaseTool) and len(t.description) >= 20 for t in tools)
    assert get_tools() is not get_tools(), "a fresh list each call, so callers cannot change each other's tools"


# =========================================================================== the tool loop in the engine


class Scripted(GenericFakeChatModel):
    """Plays back prepared AIMessages and records every prompt it was shown."""

    seen: list[list[BaseMessage]] = []
    bound: list[str] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> Scripted:  # type: ignore[override]
        self.bound.extend(t.name for t in tools)
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def tool_call(name: str, args: dict[str, Any], call_id: str = "call1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def scripted(*messages: AIMessage) -> Scripted:
    model = Scripted(messages=iter(messages))
    model.seen, model.bound = [], []
    return model


def messages_of(engine: Any, question: str = "hi") -> list[BaseMessage]:
    state = engine.graph.invoke({"messages": [HumanMessage(content=question)], "context": "", "sources": []})
    return state["messages"]  # type: ignore[no-any-return]


def test_tools_are_bound_to_the_model() -> None:
    llm = scripted(AIMessage(content="ok"))
    get_engine(llm=llm, retriever=None, tools=get_tools())
    assert llm.bound == ["calculator", "current_datetime"]


def test_no_tools_means_no_tool_node_and_an_unbound_model() -> None:
    llm = scripted(AIMessage(content="ok"))
    engine = get_engine(llm=llm, retriever=None, tools=[])
    assert isinstance(engine, LangGraphEngine)
    assert llm.bound == [] and "tools" not in engine.graph.get_graph().nodes


def test_graph_shape_with_tools() -> None:
    engine = get_engine(llm=scripted(), retriever=None, tools=get_tools())
    graph = engine.graph.get_graph()
    assert {"retrieve", "generate", "tools"} <= set(graph.nodes)
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("tools", "generate") in edges, "tool results go back to the model"
    assert ("tools", "retrieve") not in edges, "retrieval is not repeated for every tool round"


def test_the_model_asks_for_a_tool_the_tool_runs_then_the_model_answers() -> None:
    llm = scripted(tool_call("calculator", {"expression": "1847 * 392 + 15"}), AIMessage(content="It is 724,039."))
    engine = get_engine(llm=llm, retriever=None, tools=get_tools())
    messages = messages_of(engine, "What is 1847 * 392 + 15?")
    assert [type(m).__name__ for m in messages] == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    tool_message = messages[2]
    assert isinstance(tool_message, ToolMessage) and tool_message.content == "724039"
    assert tool_message.tool_call_id == "call1", "the result is matched to the request"
    assert messages[-1].text == "It is 724,039."
    last_prompt = llm.seen[-1]
    assert isinstance(last_prompt[-1], ToolMessage), "the second model call is shown the tool result"


def test_engine_invoke_returns_the_final_answer_not_the_tool_output() -> None:
    llm = FakeChatModel(reply="DONE", tool_name="calculator", tool_args={"expression": "6 * 7"})
    result = get_engine(llm=llm, retriever=None, tools=get_tools()).invoke(f"{TOOL_TRIGGER} 6*7?", [], "s1")
    assert result.answer == "DONE | tool said: 42"
    assert len(llm.calls) == 2, "exactly two model calls: ask for the tool, then answer"


def test_the_clock_tool_runs_through_the_engine(frozen_clock: None) -> None:
    llm = scripted(tool_call("current_datetime", {"timezone": "Asia/Tokyo"}), AIMessage(content="13:30"))
    messages = messages_of(get_engine(llm=llm, retriever=None, tools=get_tools()))
    assert messages[2].content == "2026-11-12T13:30:15+09:00"


def test_tool_errors_go_back_to_the_model_which_can_recover() -> None:
    llm = scripted(
        tool_call("calculator", {"expression": "__import__('os')"}),
        AIMessage(content="I can only do arithmetic."),
    )
    messages = messages_of(get_engine(llm=llm, retriever=None, tools=get_tools()))
    assert str(messages[2].content).startswith("Error") and messages[-1].text == "I can only do arithmetic."


def test_several_tool_calls_in_one_message_all_run() -> None:
    calls = AIMessage(
        content="",
        tool_calls=[
            {"name": "calculator", "args": {"expression": "2+2"}, "id": "a"},
            {"name": "calculator", "args": {"expression": "3*3"}, "id": "b"},
            {"name": "current_datetime", "args": {"timezone": "UTC"}, "id": "c"},
        ],
    )
    messages = messages_of(
        get_engine(llm=scripted(calls, AIMessage(content="done")), retriever=None, tools=get_tools())
    )
    results = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    assert results["a"] == "4" and results["b"] == "9" and str(results["c"]).endswith("+00:00")


def test_several_rounds_in_a_row() -> None:
    llm = scripted(
        tool_call("calculator", {"expression": "2+2"}, "c1"),
        tool_call("calculator", {"expression": "4*5"}, "c2"),
        AIMessage(content="20"),
    )
    messages = messages_of(get_engine(llm=llm, retriever=None, tools=get_tools()))
    assert [m.content for m in messages if isinstance(m, ToolMessage)] == ["4", "20"] and messages[-1].text == "20"


def test_a_tool_the_model_invented_does_not_crash_the_turn() -> None:
    llm = scripted(tool_call("teleport", {"where": "mars"}), AIMessage(content="I cannot do that."))
    messages = messages_of(get_engine(llm=llm, retriever=None, tools=get_tools()))
    assert "not a valid tool" in str(messages[2].content) and messages[-1].text == "I cannot do that."


def test_a_tool_that_crashes_is_reported_to_the_model_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    @tool
    def explode(text: str) -> str:
        """A tool that always crashes, for testing error handling."""
        raise RuntimeError("secret internal detail")

    llm = scripted(tool_call("explode", {"text": "x"}), AIMessage(content="Sorry, that failed."))
    with caplog.at_level(logging.ERROR, logger=engine_module.logger.name):
        messages = messages_of(get_engine(llm=llm, retriever=None, tools=[explode]))
    told_to_model = str(messages[2].content)
    assert told_to_model.startswith("Error: the tool failed (RuntimeError)")
    assert "secret internal detail" not in told_to_model, "internals are not passed on to the model or the user"
    assert "secret internal detail" in caplog.text, "but they are logged for the developer"
    assert messages[-1].text == "Sorry, that failed."


def test_retrieval_runs_once_per_turn_even_with_tool_rounds() -> None:
    retriever = StaticRetriever([make_chunk("Annual leave is 24 days.", source="leave.md")])
    llm = scripted(
        tool_call("calculator", {"expression": "24 + 5"}, "c1"),
        tool_call("calculator", {"expression": "29 * 2"}, "c2"),
        AIMessage(content="58"),
    )
    result = get_engine(llm=llm, retriever=retriever, tools=get_tools()).invoke("leave plus carry over?", [], "s1")
    assert retriever.queries == ["leave plus carry over?"]
    assert [s.source for s in result.sources] == ["leave.md"], "sources survive the tool loop"
    assert all("Annual leave is 24 days." in str(prompt[0].content) for prompt in llm.seen), (
        "every model call has the context"
    )


def test_history_still_reaches_the_model_when_tools_are_on() -> None:
    llm = scripted(AIMessage(content="Priya"))
    history = [Message(role="user", content="My name is Priya"), Message(role="assistant", content="Hi Priya")]
    get_engine(llm=llm, retriever=None, tools=get_tools()).invoke("What is my name?", history, "s1")
    assert "My name is Priya" in "\n".join(str(m.content) for m in llm.seen[-1])


# ---- runaway protection


def endless_tool_calls() -> Iterator[AIMessage]:
    number = 0
    while True:
        number += 1
        yield tool_call("calculator", {"expression": "1+1"}, f"call{number}")


def counting_calculator(counter: list[str]) -> BaseTool:
    @tool
    def calculator(expression: str) -> str:
        """Count how often the model calls this tool, and return 2."""
        counter.append(expression)
        return "2"

    return calculator


def test_a_model_that_never_stops_calling_tools_is_cut_off_quickly(caplog: pytest.LogCaptureFixture) -> None:
    runs: list[str] = []
    llm = Scripted(messages=endless_tool_calls())
    llm.seen, llm.bound = [], []
    started = time.perf_counter()
    with caplog.at_level(logging.WARNING, logger=engine_module.logger.name):
        result = get_engine(llm=llm, retriever=None, tools=[counting_calculator(runs)]).invoke("loop", [], "s9")
    assert time.perf_counter() - started < 5, "LangGraph's own limit (about 10,000 steps) takes half a minute"
    assert result.answer == TOO_MANY_STEPS_ANSWER and result.sources == []
    assert len(runs) <= MAX_TOOL_ROUNDS + 1, "a runaway costs a handful of calls, not thousands"
    assert "tool rounds" in caplog.text and "s9" in caplog.text


@pytest.mark.parametrize("rounds", [1, MAX_TOOL_ROUNDS - 1, MAX_TOOL_ROUNDS])
def test_up_to_the_maximum_number_of_rounds_still_gets_an_answer(rounds: int) -> None:
    runs: list[str] = []
    calls = [tool_call("calculator", {"expression": f"{i}+1"}, f"c{i}") for i in range(rounds)]
    llm = scripted(*calls, AIMessage(content="finished"))
    answer = get_engine(llm=llm, retriever=None, tools=[counting_calculator(runs)]).invoke("go", [], "s1").answer
    assert answer == "finished" and len(runs) == rounds


def test_one_round_over_the_maximum_is_stopped() -> None:
    runs: list[str] = []
    calls = [tool_call("calculator", {"expression": "1+1"}, f"c{i}") for i in range(MAX_TOOL_ROUNDS + 1)]
    llm = scripted(*calls, AIMessage(content="never reached"))
    answer = get_engine(llm=llm, retriever=None, tools=[counting_calculator(runs)]).invoke("go", [], "s1").answer
    assert answer == TOO_MANY_STEPS_ANSWER


def test_the_limit_is_what_the_comment_claims() -> None:
    assert RECURSION_LIMIT == 3 + 2 * MAX_TOOL_ROUNDS and RECURSION_LIMIT < 100


# ---- defaults


def test_default_tools_are_the_real_ones() -> None:
    engine = get_engine(llm=scripted(), retriever=None)
    assert isinstance(engine, LangGraphEngine)
    assert [t.name for t in engine.tools] == ["calculator", "current_datetime"]


def test_explicit_empty_tools_are_respected() -> None:
    engine = get_engine(llm=scripted(), retriever=None, tools=[])
    assert isinstance(engine, LangGraphEngine) and engine.tools == []


# ---- tracing and web


def test_tool_runs_are_traced_as_tool_runs() -> None:
    llm = scripted(tool_call("calculator", {"expression": "6*7"}), AIMessage(content="42"))
    engine = get_engine(llm=llm, retriever=None, tools=get_tools())
    with collect_runs() as collector:
        engine.invoke("6*7?", [], "session-tools")
    runs = collector.traced_runs
    tool_runs = [r for r in runs if r.run_type == "tool"]
    assert [r.name for r in tool_runs] == ["calculator"]
    assert tool_runs[0].extra["metadata"]["session_id"] == "session-tools"
    assert len([r for r in runs if r.run_type == "llm"]) == 2


@pytest.fixture()
def tools_client(client: TestClient) -> TestClient:
    llm = FakeChatModel(reply="The answer is", tool_name="calculator", tool_args={"expression": "6 * 7"})
    components = Components(engine=get_engine(llm=llm, retriever=None, tools=get_tools()))
    client.app.dependency_overrides[get_components] = lambda: components  # type: ignore[attr-defined]
    return client


def test_chat_over_http_uses_tools_and_saves_only_the_final_answer(tools_client: TestClient) -> None:
    sid = tools_client.post("/api/sessions").json()["id"]
    turn = tools_client.post(f"/api/sessions/{sid}/messages", json={"content": f"{TOOL_TRIGGER} what is 6*7?"}).json()
    assert turn["assistant_message"]["content"] == "The answer is | tool said: 42"
    saved = tools_client.get(f"/api/sessions/{sid}").json()["messages"]
    assert [m["role"] for m in saved] == ["user", "assistant"], "tool messages are not stored in the session"


def test_a_runaway_tool_loop_shows_a_friendly_message_in_the_chat(client: TestClient) -> None:
    runs: list[str] = []
    llm = Scripted(messages=endless_tool_calls())
    llm.seen, llm.bound = [], []
    components = Components(engine=get_engine(llm=llm, retriever=None, tools=[counting_calculator(runs)]))
    client.app.dependency_overrides[get_components] = lambda: components  # type: ignore[attr-defined]
    sid = client.post("/api/sessions").json()["id"]
    reply = client.post(f"/api/sessions/{sid}/messages", json={"content": "loop forever"}).json()["assistant_message"]
    assert reply["content"] == TOO_MANY_STEPS_ANSWER and not reply["content"].startswith("Engine error")
