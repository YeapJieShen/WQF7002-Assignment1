"""The Gradio app: a chat, the task list, the call log and a confirmation card for one user per tab.

The handlers are plain functions over a `Session`, so tests drive them without a browser.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import gradio as gr

from planner.adapters.config import Settings
from planner.adapters.data import load_base, load_persona, load_rules
from planner.adapters.llm import BACKEND_ERRORS, build_model
from planner.adapters.session import Reply, Session

if TYPE_CHECKING:
    from collections.abc import Callable

    from planner.app.calls import CallRecord
    from planner.domain.models import SessionState

type History = list[dict[str, str]]
type Rows = list[list[str]]

PERSONA: Final = "persona_wednesday"
BACKEND_DOWN: Final = "No language-model backend answered. Check that it is running."
SESSION_LOST: Final = "This tab's session was lost, likely by a server restart. Reload the page."
TASK_HEADERS: Final = ["ID", "Title", "Type", "Min", "Place", "Deadline", "Status"]
LOG_HEADERS: Final = ["Stage", "Tool", "Args", "Outcome"]


def default_session() -> Session:
    """Start a conversation with the configured model and the sample persona."""
    return Session(build_model(Settings()), load_base(), load_rules(), load_persona(PERSONA))


def task_rows(state: SessionState) -> Rows:
    """Describe each task as a table row."""
    return [
        [
            t.id,
            t.title,
            t.type.value,
            "" if t.duration_min is None else str(t.duration_min),
            t.place,
            f"{t.deadline:%a %m-%d %H:%M}" if t.deadline else "",
            t.status.value,
        ]
        for t in state.tasks
    ]


def log_rows(log: list[CallRecord]) -> Rows:
    """Describe each logged stage as a table row."""
    return [[r.stage, r.tool, r.args, r.outcome] for r in log]


def start(make_session: Callable[[], Session]) -> tuple[Session, Rows, Rows]:
    """Create this tab's session and show its starting tasks."""
    session = make_session()
    return session, task_rows(session.state), []


def send(text: str, history: History, session: Session | None) -> tuple[Any, ...]:
    """Pass a user message to the agent and show what came back."""
    if not text.strip():
        return tuple(gr.skip() for _ in range(6))
    live = _live(session)
    history = [*history, {"role": "user", "content": text}]
    return _render(live, history, _ask(lambda: live.send(text)))


def answer(*, approved: bool, history: History, session: Session | None) -> tuple[Any, ...]:
    """Pass the user's yes or no on a pending confirmation to the agent."""
    live = _live(session)
    return _render(live, history, _ask(lambda: live.resume(approved=approved)))


def _live(session: Session | None) -> Session:
    # Gradio keeps the session in server memory, so a restart (or hot reload) loses it.
    if session is None:
        raise gr.Error(SESSION_LOST)
    return session


def _ask(call: Callable[[], Reply]) -> Reply:
    try:
        return call()
    except BACKEND_ERRORS as exc:
        raise gr.Error(BACKEND_DOWN) from exc


def _render(session: Session, history: History, reply: Reply) -> tuple[Any, ...]:
    asking = reply.confirm is not None
    if reply.text:
        history = [*history, {"role": "assistant", "content": reply.text}]
    return (
        history,
        gr.update(value="", interactive=not asking),
        task_rows(session.state),
        log_rows(session.call_log),
        gr.update(visible=asking),
        reply.confirm or "",
    )


def build_app(make_session: Callable[[], Session] = default_session) -> gr.Blocks:
    """Lay out the app; each browser tab gets its own session from `make_session`."""
    app = gr.Blocks(title="Planning delegate")
    with app:
        session = gr.State()
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Planning delegate")
                with gr.Group(visible=False) as card:
                    question = gr.Markdown()
                    with gr.Row():
                        yes = gr.Button("Yes, do it", variant="stop")
                        no = gr.Button("No, keep it")
                box = gr.Textbox(show_label=False, placeholder="Ask me to plan something")
            with gr.Column(scale=2):
                tasks = gr.Dataframe(
                    headers=TASK_HEADERS, label="Tasks", interactive=False, wrap=True
                )
                log = gr.Dataframe(
                    headers=LOG_HEADERS, label="Call log", interactive=False, wrap=True
                )
        shown = [chat, box, tasks, log, card, question]
        app.load(lambda: start(make_session), None, [session, tasks, log])
        box.submit(send, [box, chat, session], shown)
        yes.click(lambda h, s: answer(approved=True, history=h, session=s), [chat, session], shown)
        no.click(lambda h, s: answer(approved=False, history=h, session=s), [chat, session], shown)
    return app
