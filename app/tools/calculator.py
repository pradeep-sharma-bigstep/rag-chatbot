"""Calculator tool: exact arithmetic for the model, with no `eval` anywhere.

The expression is parsed with `ast` and only a small whitelist of node types is evaluated:
numbers, `+ - * / // % **`, unary `+ -` and parentheses. Names, calls, attributes, strings,
comparisons, lambdas and everything else are rejected, so there is nothing to escape into.

Resource limits keep a hostile or careless expression (`9**9**9`) from freezing the server:
the input length, the exponent and the size of integer results are all capped.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 1_000
MAX_INT_BITS = 10_000  # about 3000 digits

Number = int | float


class CalculatorError(ValueError):
    """A problem with the expression itself; its message is shown to the model."""


def _power(base: Number, exponent: Number) -> Number:
    if abs(exponent) > MAX_EXPONENT:
        raise CalculatorError(f"exponent too large (limit {MAX_EXPONENT})")
    if (
        isinstance(base, int)
        and isinstance(exponent, int)
        and exponent > 0
        and exponent * base.bit_length() > MAX_INT_BITS
    ):
        raise CalculatorError("result too large")  # refused before computing it
    return base**exponent


_BINARY: dict[type[ast.operator], Callable[[Any, Any], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _power,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Number]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_ALLOWED = "only numbers and + - * / // % ** ( ) are allowed (use ** for powers, 0.15 * x for 15% of x)"


def _checked(value: Any) -> Number:
    """Reject results that are not an ordinary finite real number of reasonable size."""
    if isinstance(value, complex):
        raise CalculatorError("result is not a real number")
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculatorError("result is too large or undefined")
    if isinstance(value, int) and value.bit_length() > MAX_INT_BITS:
        raise CalculatorError("result too large")
    return value  # type: ignore[no-any-return]


def _evaluate(node: ast.AST) -> Number:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):  # type(): bool is excluded on purpose
        return _checked(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _checked(_UNARY[type(node.op)](_evaluate(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _checked(_BINARY[type(node.op)](_evaluate(node.left), _evaluate(node.right)))
    raise CalculatorError(_ALLOWED)


def _format(value: Number) -> str:
    return str(value) if isinstance(value, int) else format(value, ".12g")  # 0.1 + 0.2 -> "0.3"


def calculate(expression: str) -> str:
    """Evaluate `expression`; returns the result as text, or a string starting with "Error"."""
    if not isinstance(expression, str) or not expression.strip():
        return "Error: expression must be a non-empty string such as '2 + 3 * 4'"
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return f"Error: expression too long (limit {MAX_EXPRESSION_LENGTH} characters)"
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return _format(_evaluate(tree))
    except CalculatorError as exc:
        return f"Error: {exc}"
    except ZeroDivisionError:
        return "Error: division by zero"
    except OverflowError:
        return "Error: result too large"
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return f"Error: not a valid arithmetic expression; {_ALLOWED}"


_ARGUMENT_ERROR = "Error: the calculator needs one argument, `expression`, as text such as '2 + 3 * 4'"


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly and return the result as text.

    Use this for every calculation instead of working it out yourself. Supports numbers,
    + - * / // % ** and parentheses, for example "(12.5 + 3) * 4" or "2 ** 10".
    Write powers with ** (not ^) and percentages as multiplication: 15% of 240 is "0.15 * 240".
    Returns "Error: ..." with the reason when the expression is invalid.
    """
    return calculate(expression)


calculator.handle_validation_error = _ARGUMENT_ERROR  # a wrong argument type must also come back as "Error: ..."
