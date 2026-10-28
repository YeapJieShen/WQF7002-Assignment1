"""One conversation with the agent: send a message, answer a confirmation, read the session."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from planner.adapters.graph import DEFAULT_MAX_REJECTIONS, build_graph

if TYPE_CHECKING:
    from datetime import datetime

    from planner.adapters.graph import Model
    from planner.app.calls import CallRecord
    from planner.domain.models import BaseData, Pace, SessionState


@dataclass(frozen=True)
class Reply:
    """What the agent sent back after a turn.

    Attributes:
        text: The assistant's message; empty while a confirmation is pending.
        confirm: A yes/no question the user must answer through `Session.resume`, if any.
    """

    text: str
    confirm: str | None = None


class Session:
    """A conversation that starts from one session state and keeps its own history."""

    def __init__(
        self,
        model: Model,
        base: BaseData,
        rules: str,
        start: SessionState,
        *,
        max_rejections: int = DEFAULT_MAX_REJECTIONS,
    ) -> None:
        """Create a conversation.

        Args:
            model: Produces the assistant messages.
            base: The read-only reference data.
            rules: The fixed system prompt.
            start: The session state to begin from, such as a loaded persona.
            max_rejections: Rejected calls in a row before the agent gives up.
        """
        self._base = base
        self._graph = build_graph(model, base, rules, max_rejections=max_rejections)
        self._config: Any = {"configurable": {"thread_id": uuid4().hex}}
        self._start: SessionState | None = start

    @property
    def state(self) -> SessionState:
        """The session as the agent's tools have left it."""
        if self._start is not None:
            return self._start
        return self._graph.get_state(self._config).values["session"]  # type: ignore[no-any-return]

    @property
    def call_log(self) -> list[CallRecord]:
        """Every stage every tool call has passed through, oldest first."""
        return list(self._graph.get_state(self._config).values.get("call_log", []))

    def update_session(
        self,
        *,
        now: datetime | None = None,
        location: str | None = None,
        pace: Pace | None = None,
    ) -> None:
        """Change the clock, location or pace; the model has no tool for these.

        Args:
            now: The new simulated time.
            location: The new place id.
            pace: The new pace settings.

        Raises:
            ValueError: If the location is not a known place.
            RuntimeError: If a confirmation is waiting for an answer.
        """
        if location is not None and location not in self._base.place_ids:
            msg = f"unknown place: {location}"
            raise ValueError(msg)
        if self._start is None and self._graph.get_state(self._config).next:
            msg = "answer the pending confirmation first"
            raise RuntimeError(msg)
        changes = {"now": now, "location": location, "pace": pace}
        updated = self.state.model_copy(update={k: v for k, v in changes.items() if v is not None})
        if self._start is not None:
            self._start = updated
        else:
            self._graph.update_state(self._config, {"session": updated})

    def send(self, text: str) -> Reply:
        """Send a user message and run the agent until it replies or needs a confirmation."""
        # Any: later turns send only the new message, a partial update of the full AgentState.
        message: Any = {"messages": [HumanMessage(text)]}
        if self._start is not None:
            message = {**message, "session": self._start, "pending": None, "rejections": 0}
            self._start = None
        return _reply(self._graph.invoke(message, self._config))

    def resume(self, approved: bool) -> Reply:  # noqa: FBT001 - one yes/no answer to the pending question
        """Answer the pending confirmation: True runs the call, False cancels it."""
        return _reply(self._graph.invoke(Command(resume=approved), self._config))


def _reply(result: dict[str, Any]) -> Reply:
    if result.get("__interrupt__"):
        return Reply("", confirm=result["__interrupt__"][0].value)
    return Reply(result["messages"][-1].text)
