"""The chat model: one client per endpoint, chained so the next one answers when a backend fails."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import openai
from langchain_openai import ChatOpenAI

from planner.app.tools.args import TOOLS_BY_NAME

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import Runnable

    from planner.adapters.config import Endpoint, Settings

# Backend trouble only. A bad answer is the graph's job (retry cap), not a reason to switch.
BACKEND_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)

type ChatRunnable = Runnable[LanguageModelInput, AIMessage]


def tool_specs() -> list[dict[str, Any]]:
    """Describe the seven tools in the OpenAI function format, named by `tool_name`."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": inspect.getdoc(args) or "",
                "parameters": args.model_json_schema(),
            },
        }
        for name, args in TOOLS_BY_NAME.items()
    ]


def build_model(settings: Settings) -> ChatRunnable:
    """Build the chat model for the configured endpoints, best first.

    Args:
        settings: Runtime settings with at least one endpoint.

    Returns:
        A model that moves to the next endpoint on connection, timeout, rate-limit or 5xx errors.
    """
    return chain([_client(e) for e in settings.llm_endpoints])


def chain(models: Sequence[ChatRunnable]) -> ChatRunnable:
    """Try the models in order, moving on only after a backend error."""
    first, *rest = models
    if not rest:
        return first
    return first.with_fallbacks(list(rest), exceptions_to_handle=BACKEND_ERRORS)


def _client(endpoint: Endpoint) -> ChatRunnable:
    # max_retries=0: a failing endpoint should hand over to the next one at once.
    chat = ChatOpenAI(
        openai_api_base=endpoint.base_url,
        model_name=endpoint.model,
        openai_api_key=endpoint.api_key,
        request_timeout=endpoint.timeout_s,
        max_retries=0,
    )
    return chat.bind_tools(tool_specs(), parallel_tool_calls=False)
