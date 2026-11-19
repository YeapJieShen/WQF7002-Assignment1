"""One browser tab's conversation, without Gradio: the server-side state and the streamed turn.

A `Tab` holds everything the page must not carry back and forth: the session, the visitor's own
endpoint (key included) and the transcript so far. The streamed turn turns session events into
chat messages, so tests drive it without a browser.
"""

from __future__ import annotations

import contextvars
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit
from uuid import uuid4

from planner.adapters.logs import activate, bind, deactivate, elapsed_ms, emit, forget_secret
from planner.adapters.session import Done, Event, Session, Stage, Token
from planner.adapters.transcript import Turn
from planner.adapters.views import Message, call_message, thinking, usage_footer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from langchain_core.messages import AnyMessage

    from planner.adapters.backend import Backend
    from planner.adapters.config import Endpoint
    from planner.adapters.graph import Model
    from planner.app.calls import CallRecord
    from planner.domain.models import BaseData, SessionState

type History = list[Message]

WAITING = "waiting for you"
LOG = logging.getLogger("planner.chat")
BACKEND_LOG = logging.getLogger("planner.backends")


class SwitchableModel:
    """A model whose backend can change mid-conversation, because the graph is built only once."""

    def __init__(self, model: Model, custom: Endpoint | None = None) -> None:
        """Start with `model` answering; `custom` says who it is, for the logs."""
        self.model = model
        self.source = _source(custom)

    def invoke(self, messages: Sequence[AnyMessage], /) -> Any:  # noqa: ANN401 - the model's AIMessage
        """Ask whichever model is current, logging how long it took and what it used."""
        started = time.perf_counter()
        try:
            reply = self.model.invoke(messages)
        except Exception as exc:
            emit(
                BACKEND_LOG,
                "backend_error",
                level=logging.WARNING,
                **self.source,
                duration_ms=elapsed_ms(started),
                error_class=type(exc).__name__,
                status_code=getattr(exc, "status_code", None),
            )
            raise
        usage: dict[str, Any] = dict(reply.usage_metadata or {})
        emit(
            BACKEND_LOG,
            "backend_call",
            **self.source,
            duration_ms=elapsed_ms(started),
            tokens_in=usage.get("input_tokens"),
            tokens_out=usage.get("output_tokens"),
            tool_calls=len(reply.tool_calls) + len(reply.invalid_tool_calls),
        )
        return reply


def _source(custom: Endpoint | None) -> dict[str, Any]:
    """Who answers, for the logs: the host and model name of a custom endpoint, never its key."""
    if custom is None:
        return {"backend": "default"}
    return {"backend": "custom", "host": urlsplit(custom.base_url).hostname, "model": custom.model}


@dataclass
class Tab:
    """What the server remembers for one browser tab.

    Attributes:
        session: The conversation with the agent.
        switch: The model slot the session's graph calls.
        custom: The visitor's own endpoint, key included; never leaves the server.
        real_clock: Whether the session follows the real clock rather than a simulated one.
        turns: Every turn so far, for the transcript.
        pending: The confirmation question waiting for an answer, if any.
    """

    session: Session
    switch: SwitchableModel
    custom: Endpoint | None = None
    real_clock: bool = True
    turns: list[Turn] = field(default_factory=list)
    pending: str | None = None


def new_tab(  # noqa: PLR0913 - one argument per thing a tab starts from
    backend: Backend,
    base: BaseData,
    rules: str,
    state: SessionState,
    *,
    real_clock: bool,
    max_rejections: int,
    custom: Endpoint | None = None,
) -> Tab:
    """Start a tab's conversation from `state`, answered by the default model or `custom`.

    Raises:
        ValueError: If the backend does not accept `custom`.
    """
    switch = SwitchableModel(backend.model(custom), custom)
    return Tab(
        Session(switch, base, rules, state, max_rejections=max_rejections),
        switch,
        custom,
        real_clock,
    )


def use_endpoint(tab: Tab, backend: Backend, custom: Endpoint | None) -> None:
    """Answer from now on with `custom`, or with the default model when it is None.

    Raises:
        ValueError: If the backend does not accept `custom`; the tab is left as it was.
    """
    tab.switch.model = backend.model(custom)  # registers the new key with the logs
    tab.switch.source = _source(custom)
    release_endpoint(tab.custom)
    tab.custom = custom


def release_endpoint(custom: Endpoint | None) -> None:
    """Stop scrubbing `custom`'s key from the logs, once nothing else uses it."""
    if custom is not None:
        forget_secret(custom.api_key.get_secret_value())


def stream_turn(  # noqa: PLR0913 - the turn's collaborators
    tab: Tab,
    backend: Backend,
    history: History,
    events: Callable[[], Iterator[Event]],
    *,
    said: str,
    signed_in: bool,
    rotate_s: float,
) -> Iterator[History]:
    """Run one turn and yield the chat after each token and each call stage.

    Tokens grow an assistant bubble; each call stage becomes its own row. A backend failure that
    the backend can explain becomes a message; any other failure is raised.

    Args:
        tab: The tab the turn belongs to; its transcript gains the turn.
        backend: Explains failures and prices the turn.
        history: The chat so far, which already ends with the user's message if there is one.
        events: Starts the session's stream; called inside the turn so failures are caught.
        said: What the user said or answered, for the transcript.
        signed_in: Whether the visitor is signed in, which changes how a failure is explained.
        rotate_s: Seconds of silence before the pending line changes its text.

    Yields:
        The whole chat each time it changes.
    """
    turn = Turn(said=said)
    started = time.perf_counter()
    shown = _Shown(list(history))
    if tab.turns and tab.turns[-1].confirm:
        shown.resume(tab.turns[-1].stages)
    turn_id, session = uuid4().hex, tab.session.log_id
    outcome = "error"
    activate(turn_id, session)
    emit(LOG, "turn_start", turns_before=len(tab.turns), signed_in=signed_in, **tab.switch.source)
    if LOG.isEnabledFor(logging.DEBUG):  # what the user said is user data: DEBUG only
        emit(LOG, "turn_input", level=logging.DEBUG, text=said)
    try:
        yield shown.frame()
        for event in _heartbeat(bind(turn_id, session, events()), rotate_s):
            match event:
                case Tick():
                    shown.tick += 1
                case Token(text):
                    shown.token(text)
                case Stage(record):
                    turn.stages.append(record)
                    shown.stage(record)
                case Done(reply, usage):
                    turn.reply, turn.confirm, turn.usage = reply.text, reply.confirm, usage
                    tab.pending = reply.confirm
                    turn.cost = backend.cost(usage, tab.custom) if usage else None
                    footer = usage_footer(time.perf_counter() - started, usage)
                    shown.done(
                        reply.text, waiting=reply.confirm is not None, note=turn.cost, footer=footer
                    )
                    outcome = "confirm_pending" if reply.confirm else "reply"
            yield shown.frame()
    except Exception as exc:
        activate(turn_id, session)
        note = backend.explain(exc, signed_in=signed_in, custom=tab.custom is not None)
        if note is None:
            emit(
                LOG, "turn_error", level=logging.ERROR, exc_info=exc, error_class=type(exc).__name__
            )
            raise
        outcome = "backend_error"
        turn.error = note
        shown.chat.append({"role": "assistant", "content": note})
        shown.busy = False
        yield shown.frame()
    finally:
        turn.seconds = time.perf_counter() - started
        tab.turns.append(turn)
        activate(turn_id, session)
        _log_turn_end(turn, outcome)
        deactivate()


@dataclass
class _Shown:
    """The chat as the page shows it during a turn: bubbles and one row per tool call."""

    chat: History
    open_bubble: bool = False  # whether the last message is a reply still growing token by token
    calls: list[CallRecord] = field(default_factory=list)  # the stages of the call in progress
    row: int = -1  # where that call's row sits in the chat
    busy: bool = True  # nothing to read yet, so the page shows the pending line
    tool: str | None = None  # the call about to run, named in the pending line
    tick: int = 0

    def resume(self, stages: list[CallRecord]) -> None:
        """Pick up the row of the call that was waiting for the user's answer, to update it."""
        # The browser sends `metadata: None` for plain messages, so a missing key is not enough.
        waiting = [
            i
            for i, m in enumerate(self.chat)
            if WAITING in (m.get("metadata") or {}).get("title", "")
        ]
        if waiting:
            first = max(i for i, r in enumerate(stages) if r.stage == "check")
            self.calls, self.row = stages[first:], waiting[-1]

    def frame(self) -> History:
        """The chat to show now, with the pending line last while there is nothing to read."""
        if self.busy:
            return [*self.chat, thinking(self.tick, self.tool)]
        return list(self.chat)

    def token(self, text: str) -> None:
        """Grow the reply bubble, opening it first if the last message is not one."""
        self.busy = False
        if self.open_bubble:
            self.chat[-1] = {**self.chat[-1], "content": self.chat[-1]["content"] + text}
        else:
            self.chat.append({"role": "assistant", "content": text})
            self.open_bubble = True

    def stage(self, record: CallRecord) -> None:
        """Start the call's row at its check and update it in place at each later stage."""
        if record.stage == "check" or not self.calls:
            self.calls, self.row = [], len(self.chat)
            self.chat.append({})
        self.calls.append(record)
        self.chat[self.row] = call_message(self.calls)
        self.open_bubble = False
        self.busy = True
        self.tool = record.tool if _still_running(record) else None

    def done(self, text: str, *, waiting: bool, note: str | None, footer: str) -> None:
        """Settle the last bubble to the reply, marking the call's row if it awaits an answer.

        A reply ends with its cost note and a footer of the turn's seconds and tokens.
        """
        self.busy = False
        if waiting and self.calls:
            self.chat[self.row] = call_message(self.calls, waiting=True)
        self.chat = _settle(self.chat, text, open_bubble=self.open_bubble)
        if text:
            extra = f"\n\n*{note}*" if note else ""
            content = f"{self.chat[-1]['content']}{extra}\n\n{footer}"
            self.chat[-1] = {**self.chat[-1], "content": content}


def _still_running(record: CallRecord) -> bool:
    """Whether the call is accepted or approved and so about to run, as opposed to over."""
    return record.outcome in {"accepted", "approved"}


@dataclass(frozen=True)
class Tick:
    """No event arrived for a while: time to rotate the pending text."""


def _heartbeat(events: Iterator[Event], every: float) -> Iterator[Event | Tick]:
    """Yield `events`, plus a `Tick` each time `every` seconds pass without one.

    The events are produced on a helper thread (in a copy of this context, so logs stay tagged)
    because a blocked model call cannot yield; if the consumer stops early, the thread ends after
    its current event.
    """
    box: queue.Queue[Event | BaseException | None] = queue.Queue()
    stop = threading.Event()

    def pump() -> None:
        try:
            for event in events:
                if stop.is_set():
                    return
                box.put(event)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the consumer's thread
            box.put(exc)
        finally:
            box.put(None)

    threading.Thread(target=contextvars.copy_context().run, args=(pump,), daemon=True).start()
    try:
        while True:
            try:
                item = box.get(timeout=every)
            except queue.Empty:
                yield Tick()
                continue
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()


def _log_turn_end(turn: Turn, outcome: str) -> None:
    emit(
        LOG,
        "turn_end",
        outcome=outcome,
        latency_ms=round(turn.seconds * 1000, 2),
        tokens_in=turn.usage.input_tokens if turn.usage else None,
        tokens_out=turn.usage.output_tokens if turn.usage else None,
        n_tool_calls=sum(1 for r in turn.stages if r.stage == "check"),
    )
    if turn.reply and LOG.isEnabledFor(logging.DEBUG):  # the reply echoes user data: DEBUG only
        emit(LOG, "turn_reply", level=logging.DEBUG, text=turn.reply)


def _settle(chat: History, text: str, *, open_bubble: bool) -> History:
    """Make the last bubble the reply's final text, whether or not its tokens were streamed."""
    if not text:
        return chat
    if open_bubble:
        return [*chat[:-1], {**chat[-1], "content": text}]
    return [*chat, {"role": "assistant", "content": text}]
