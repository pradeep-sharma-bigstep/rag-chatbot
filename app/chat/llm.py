"""MILESTONE 1 — The LLM.

Return a LangChain chat model for the provider/model set in `.env`.

  LLM_PROVIDER=anthropic  -> `uv add langchain-anthropic`, use ChatAnthropic
  LLM_PROVIDER=openai     -> `uv add langchain-openai`,    use ChatOpenAI

The API key is read by the provider package from the environment
(ANTHROPIC_API_KEY / OPENAI_API_KEY). Never hard-code keys.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel


def get_llm() -> BaseChatModel:
    """OUT: a ready-to-use chat model (supports `.invoke(messages)` and `.bind_tools(tools)`).

    TODO(milestone 1):
      - Read provider and model name from `app.config.get_settings()`.
      - Build and return the matching chat model; raise ValueError for an unknown provider.
      - Keep temperature low (e.g. 0) so answers are repeatable.
    """
    raise NotImplementedError("TODO(milestone 1): implement get_llm in app/chat/llm.py")
