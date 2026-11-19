# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
"""Test doubles for the model and the backend; CI never calls an LLM."""

from collections.abc import Iterator, Sequence

from langchain_core.messages import AIMessage, AnyMessage

from planner.adapters.config import Endpoint
from planner.adapters.graph import Model
from planner.adapters.session import Usage


class ScriptedModel:
    def __init__(self, *script: AIMessage) -> None:
        self._script: Iterator[AIMessage] = iter(script)

    def invoke(self, messages: Sequence[AnyMessage], /) -> AIMessage:  # noqa: ARG002 - protocol
        return next(self._script)


class Boom(Exception):  # noqa: N818 - a stand-in for a backend error
    pass


class FailingModel:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def invoke(self, messages: Sequence[AnyMessage], /) -> AIMessage:  # noqa: ARG002 - protocol
        raise self._error


def call(name: str, args: dict[str, object]) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": "c1"}])


class FakeBackend:
    """Hands out scripted models; explains `Boom` as a quota problem and prices every turn."""

    def __init__(self, default: Model, *, price: str | None = "12 tokens") -> None:
        self.default = default
        self.price = price
        self.asked: list[Endpoint | None] = []
        self.explained: list[tuple[bool, bool]] = []

    def model(self, custom: Endpoint | None, /) -> Model:
        self.asked.append(custom)
        if custom is not None and not custom.base_url.startswith("https://"):
            msg = "the URL must be https"
            raise ValueError(msg)
        return self.default if custom is None else ScriptedModel()

    def explain(self, error: Exception, /, *, signed_in: bool, custom: bool) -> str | None:
        self.explained.append((signed_in, custom))
        return "Out of quota." if isinstance(error, Boom) else None

    def cost(self, usage: Usage, custom: Endpoint | None, /) -> str | None:  # noqa: ARG002 - protocol; the fake prices every endpoint alike
        return self.price
