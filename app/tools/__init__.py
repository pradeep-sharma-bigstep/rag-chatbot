"""MILESTONE 8 — Tools the LLM can call.

Put each tool in its own module in this package (e.g. `calculator.py`) using
`from langchain_core.tools import tool`, and list it in `get_tools()`.

Required tools (the evaluator calls them directly):

  calculator(expression: str) -> str
      Evaluates arithmetic like "2 + 3 * 4" -> "14".
      Must NOT use eval()/exec() on raw input; for anything that is not plain
      arithmetic, return a string starting with "Error" instead of raising.

  current_datetime() -> str
      Current date and time in ISO 8601 format.

Add more if you like (e.g. `search_documents(query)` that uses the retriever).
Every tool needs a clear description (the docstring) — the LLM reads it to
decide when to call the tool.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool


def get_tools() -> list[BaseTool]:
    """OUT: every tool the engine should bind to the LLM.

    TODO(milestone 8): import your tools and return them, e.g. `return [calculator, current_datetime]`.
    """
    raise NotImplementedError("TODO(milestone 8): implement get_tools in app/tools/__init__.py")
