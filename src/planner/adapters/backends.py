"""The `Backend` the app runs with: the deployment's default model plus the visitor's own endpoint.

A custom endpoint is vetted by `endpoint_guard` (public https only). Its key lives only in its
`Endpoint` and the client built from it. Provider errors are re-raised as `ProviderError` with the
key cut out, so neither the chat, the logs nor a traceback can show it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import gradio as gr
import openai

from planner.adapters import zerogpu
from planner.adapters.endpoint_guard import (
    BlockedEndpointError,
    check_url,
    guarded_async_client,
    guarded_client,
)
from planner.adapters.llm import BACKEND_ERRORS, build_model, client

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.messages import AIMessage, AnyMessage

    from planner.adapters.config import Endpoint, Settings
    from planner.adapters.graph import Model
    from planner.adapters.llm import ChatRunnable
    from planner.adapters.session import Usage

ZEROGPU_DOCS: Final = "https://huggingface.co/docs/hub/spaces-zerogpu"
BILLING: Final = "https://huggingface.co/settings/billing"
BACKEND_DOWN: Final = "No language-model backend answered. Check that it is running."
REDACTED: Final = "[key removed]"


class ProviderError(Exception):
    """A custom endpoint failed; the message is the provider's, without the key."""


class EndpointTimeoutError(ProviderError):
    """A custom endpoint did not answer in time; the message says how long it was given."""


@dataclass(frozen=True)
class AppBackend:
    """Implements `Backend` for one deployment.

    Attributes:
        default: The model used when the visitor has not given an endpoint.
        zerogpu: Whether `default` runs on ZeroGPU, which has a per-visitor quota.
        allow_private: Whether a visitor's endpoint may use http and private addresses.
        error_max_chars: Longest provider error text echoed to the chat.
    """

    default: Model
    zerogpu: bool
    allow_private: bool
    error_max_chars: int

    def model(self, custom: Endpoint | None, /) -> Model:
        """Return the default model, or a client for the visitor's endpoint.

        Raises:
            BlockedEndpointError: If the endpoint is not a public https address.
        """
        if custom is None:
            return self.default
        check_url(custom.base_url, allow_private=self.allow_private)
        guarded = not self.allow_private
        return _Redacting(
            client(
                custom,
                guarded_client() if guarded else None,
                guarded_async_client() if guarded else None,
            ),
            custom.api_key.get_secret_value(),
            custom.timeout_s,
            self.error_max_chars,
        )

    def explain(self, error: Exception, /, *, signed_in: bool, custom: bool) -> str | None:
        """Return the chat message for a backend failure, or None if `error` is not one."""
        if custom:
            if isinstance(error, EndpointTimeoutError):
                return str(error)
            if isinstance(error, ProviderError):
                return f"Your endpoint returned an error: {error}"
            return None
        if self.zerogpu:
            # spaces reports quota, queue and worker failures as gr.Error.
            if not isinstance(error, gr.Error):
                return None
            if signed_in:
                hint = f"Check your remaining GPU time at {BILLING}."
            else:
                hint = "Sign in with Hugging Face for more GPU time, or use your own endpoint."
            return f"The GPU is not available: {error.message}\n\n{hint} See {ZEROGPU_DOCS}"
        return BACKEND_DOWN if isinstance(error, BACKEND_ERRORS) else None

    def cost(self, usage: Usage, custom: Endpoint | None, /) -> str | None:
        """Return the tokens a custom endpoint used, which its provider may charge for."""
        if custom is None:
            return None
        return f"{usage.input_tokens:,} in / {usage.output_tokens:,} out tokens"


@dataclass(frozen=True)
class _Redacting:
    """A custom endpoint's client whose errors never carry the key."""

    chat: ChatRunnable = field(repr=False)
    key: str = field(repr=False)
    timeout_s: float
    error_max_chars: int

    def invoke(self, messages: Sequence[AnyMessage], /) -> AIMessage:
        try:
            return self.chat.invoke(messages)
        except openai.APITimeoutError:
            msg = (
                f"Your endpoint didn't answer within {self.timeout_s:g} s. "
                "Check that it is up, or try a faster model."
            )
            raise EndpointTimeoutError(msg) from None
        except BlockedEndpointError as exc:  # refused by the guard before the request
            raise ProviderError(str(exc)) from None
        except openai.OpenAIError as exc:
            raise self._provider_error(exc) from None  # the original may quote the key

    def _provider_error(self, exc: Exception) -> ProviderError:
        """Return the provider's message without the key, cut to the configured length."""
        text = str(exc).replace(self.key, REDACTED) if self.key else str(exc)
        if len(text) > self.error_max_chars:
            text = text[: self.error_max_chars] + "..."
        return ProviderError(text)


def build_backend(settings: Settings) -> AppBackend:
    """Build the backend `settings.backend` names.

    "zerogpu" loads the model onto the GPU, so call this once while the app starts.
    """
    if settings.backend == "zerogpu":
        chat = zerogpu.TemplateChat(
            engine=zerogpu.load_engine(settings),
            call_open=settings.tool_call_open,
            call_close=settings.tool_call_close,
        )
        return AppBackend(
            chat,
            zerogpu=True,
            allow_private=settings.allow_private_endpoints,
            error_max_chars=settings.provider_error_max_chars,
        )
    return AppBackend(
        build_model(settings),
        zerogpu=False,
        allow_private=settings.allow_private_endpoints,
        error_max_chars=settings.provider_error_max_chars,
    )
