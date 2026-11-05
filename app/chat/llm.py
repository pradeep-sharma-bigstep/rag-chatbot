"""MILESTONE 1 — The LLM.

Return a LangChain chat model for the provider/model set in `.env`.

  LLM_PROVIDER=groq       -> langchain-groq (installed),        use ChatGroq
  LLM_PROVIDER=anthropic  -> `uv add langchain-anthropic` first, use ChatAnthropic
  LLM_PROVIDER=openai     -> `uv add langchain-openai` first,    use ChatOpenAI

The API key is read by the provider package from the environment
(GROQ_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY). Never hard-code keys.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

from app.config import get_settings


def _groq(model: str) -> BaseChatModel:
    # temperature=0 keeps answers repeatable (ChatGroq sends it to the API as 1e-8).
    return ChatGroq(model=model, temperature=0)


def _anthropic(model: str) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic  # requires `uv add langchain-anthropic`

    # No temperature: current Claude models reject non-default sampling values.
    return ChatAnthropic(model_name=model, max_tokens_to_sample=16_000, timeout=None, stop=None)


def _openai(model: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI  # requires `uv add langchain-openai`

    return ChatOpenAI(model=model, temperature=0)


PROVIDERS: dict[str, Callable[[str], BaseChatModel]] = {
    "groq": _groq,
    "anthropic": _anthropic,
    "openai": _openai,
}


def get_llm() -> BaseChatModel:
    """OUT: a ready-to-use chat model (supports `.invoke(messages)` and `.bind_tools(tools)`).

    Provider and model come from LLM_PROVIDER / LLM_MODEL. Raises ValueError for an
    unknown provider or a missing model name.
    """
    settings = get_settings()
    provider = settings.llm_provider.strip().lower()
    model = settings.llm_model.strip()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER {provider!r}; expected one of {sorted(PROVIDERS)}")
    if not model:
        raise ValueError("LLM_MODEL is empty; set it in .env")
    return PROVIDERS[provider](model)
