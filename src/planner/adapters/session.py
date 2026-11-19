"""One conversation with the agent: send a message, answer a confirmation, read the session.

`stream` and `resume_stream` yield events while the turn runs: reply tokens, each stage of each
tool call, then one final `Done`. `send` and `resume` run the same turns and return only the reply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.types import Command

from planner.adapters.graph import build_graph

if TYPE_CHECKING:
    from collections.abc import Iterator
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


@dataclass(frozen=True)
class Usage:
    """Tokens the model used in one turn, summed over every model call in it.

    Attributes:
        input_tokens: Prompt tokens.
        output_tokens: Generated tokens.
    """

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class Token:
    """A piece of the assistant's reply text, in order."""

    text: str


@dataclass(frozen=True)
class Stage:
    """One stage of one tool call, emitted as soon as it happens."""

    record: CallRecord


@dataclass(frozen=True)
class Done:
    """The end of a turn.

    Attributes:
        reply: What `send` or `resume` would have returned.
        usage: Tokens used, or None if the model reported none.
    """

    reply: Reply
    usage: Usage | None = None


type Event = Token | Stage | Done


class Session:
    """A conversation that starts from one session state and keeps its own history."""

    def __init__(
        self,
        model: Model,
        base: BaseData,
        rules: str,
        start: SessionState,
        *,
        max_rejections: int,
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
        if self._start is None and self._graph.get_state(self._config).interrupts:
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
        return _final(self.stream(text)).reply

    def resume(self, approved: bool) -> Reply:  # noqa: FBT001 - one yes/no answer to the pending question
        """Answer the pending confirmation: True runs the call, False cancels it."""
        return _final(self.resume_stream(approved)).reply

    def stream(self, text: str) -> Iterator[Event]:
        """Like `send`, but yield tokens and call stages as they happen, then `Done`."""
        # Any: later turns send only the new message, a partial update of the full AgentState.
        message: Any = {"messages": [HumanMessage(text)]}
        if self._start is not None:
            message = {**message, "session": self._start, "pending": None, "rejections": 0}
            self._start = None
        return self._run(message)

    def resume_stream(self, approved: bool) -> Iterator[Event]:  # noqa: FBT001 - one yes/no answer
        """Like `resume`, but yield tokens and call stages as they happen, then `Done`."""
        return self._run(Command(resume=approved))

    def _run(self, graph_input: Any) -> Iterator[Event]:  # noqa: ANN401 - a state update or a Command
        used: list[AIMessage] = []
        for item in self._graph.stream(
            graph_input, self._config, stream_mode=["messages", "updates"]
        ):
            # With a list of modes LangGraph yields (mode, chunk) pairs; its stubs do not say so.
            mode, chunk = cast("tuple[str, Any]", item)
            if mode == "messages":
                piece, meta = chunk
                # Model output only; give_up's fixed apology arrives in Done.
                if meta.get("langgraph_node") == "call_model" and piece.text:
                    yield Token(piece.text)
                continue
            for update in chunk.values():
                if not isinstance(update, dict):
                    continue  # the interrupt marker is a tuple, not a state update
                used += [m for m in update.get("messages", []) if _has_usage(m)]
                yield from (Stage(r) for r in update.get("call_log", []))
        yield Done(self._reply(), _usage(used))

    def _reply(self) -> Reply:
        snapshot = self._graph.get_state(self._config)
        if snapshot.interrupts:
            return Reply("", confirm=snapshot.interrupts[0].value)
        return Reply(snapshot.values["messages"][-1].text)


def _has_usage(message: BaseMessage) -> bool:
    return isinstance(message, AIMessage) and message.usage_metadata is not None


def _usage(messages: list[AIMessage]) -> Usage | None:
    reports = [m.usage_metadata for m in messages if m.usage_metadata is not None]
    if not reports:
        return None
    return Usage(
        input_tokens=sum(r["input_tokens"] for r in reports),
        output_tokens=sum(r["output_tokens"] for r in reports),
    )


def _final(events: Iterator[Event]) -> Done:
    *_, last = events
    if not isinstance(last, Done):  # pragma: no cover - _run always ends with Done
        msg = "a turn must end with Done"
        raise TypeError(msg)
    return last
