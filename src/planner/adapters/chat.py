"""One browser tab's conversation, without Gradio: the server-side state and the streamed turn.

A `Tab` holds everything the page must not carry back and forth: the session, the visitor's own
endpoint (key included) and the transcript so far. The streamed turn turns session events into
chat messages, so tests drive it without a browser.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from planner.adapters.session import Done, Event, Session, Stage, Token
from planner.adapters.transcript import Turn
from planner.adapters.views import Message, stage_message

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from langchain_core.messages import AnyMessage

    from planner.adapters.backend import Backend
    from planner.adapters.config import Endpoint
    from planner.adapters.graph import Model
    from planner.domain.models import BaseData, SessionState

type History = list[Message]


class SwitchableModel:
    """A model whose backend can change mid-conversation, because the graph is built only once."""

    def __init__(self, model: Model) -> None:
        """Start with `model` answering."""
        self.model = model

    def invoke(self, messages: Sequence[AnyMessage], /) -> Any:  # noqa: ANN401 - the model's AIMessage
        """Ask whichever model is current."""
        return self.model.invoke(messages)


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
    switch = SwitchableModel(backend.model(custom))
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
    tab.switch.model = backend.model(custom)
    tab.custom = custom


def stream_turn(  # noqa: PLR0913 - the turn's collaborators
    tab: Tab,
    backend: Backend,
    history: History,
    events: Callable[[], Iterator[Event]],
    *,
    said: str,
    signed_in: bool,
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

    Yields:
        The whole chat each time it changes.
    """
    turn = Turn(said=said)
    started = time.perf_counter()
    chat = list(history)
    open_bubble = False
    try:
        for event in events():
            match event:
                case Token(text):
                    if open_bubble:
                        chat[-1] = {**chat[-1], "content": chat[-1]["content"] + text}
                    else:
                        chat.append({"role": "assistant", "content": text})
                        open_bubble = True
                case Stage(record):
                    turn.stages.append(record)
                    chat.append(stage_message(record))
                    open_bubble = False
                case Done(reply, usage):
                    turn.reply, turn.confirm, turn.usage = reply.text, reply.confirm, usage
                    tab.pending = reply.confirm
                    chat = _settle(chat, reply.text, open_bubble=open_bubble)
                    turn.cost = backend.cost(usage, tab.custom) if usage else None
                    if turn.cost and reply.text:
                        chat[-1] = {
                            **chat[-1],
                            "content": f"{chat[-1]['content']}\n\n*{turn.cost}*",
                        }
            yield list(chat)
    except Exception as exc:
        note = backend.explain(exc, signed_in=signed_in, custom=tab.custom is not None)
        if note is None:
            raise
        turn.error = note
        chat.append({"role": "assistant", "content": note})
        yield list(chat)
    finally:
        turn.seconds = time.perf_counter() - started
        tab.turns.append(turn)


def _settle(chat: History, text: str, *, open_bubble: bool) -> History:
    """Make the last bubble the reply's final text, whether or not its tokens were streamed."""
    if not text:
        return chat
    if open_bubble:
        return [*chat[:-1], {**chat[-1], "content": text}]
    return [*chat, {"role": "assistant", "content": text}]
