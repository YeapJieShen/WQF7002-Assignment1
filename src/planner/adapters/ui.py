"""The Gradio app: a chat, the task list, the call log and a confirmation card for one user per tab.

The handlers are plain functions over a `Session`, so tests drive them without a browser.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

import gradio as gr

from planner.adapters.config import Settings
from planner.adapters.data import list_personas, load_base, load_persona, load_rules
from planner.adapters.llm import BACKEND_ERRORS, build_model
from planner.adapters.session import Reply, Session

if TYPE_CHECKING:
    from collections.abc import Callable

    from planner.app.calls import CallRecord
    from planner.domain.models import SessionState

type History = list[dict[str, str]]
type Rows = list[list[str]]

PERSONA: Final = "persona_wednesday"
BAD_CLOCK: Final = "The clock must look like 2026-10-28 15:40."
BACKEND_DOWN: Final = "No language-model backend answered. Check that it is running."
SESSION_LOST: Final = "This tab's session was lost, likely by a server restart. Reload the page."
TASK_HEADERS: Final = ["ID", "Title", "Type", "Min", "Place", "Deadline", "Status"]
LOG_HEADERS: Final = ["Stage", "Tool", "Args", "Outcome"]


def default_session(persona: str = PERSONA) -> Session:
    """Start a conversation with the configured model and a sample persona."""
    return Session(build_model(Settings()), load_base(), load_rules(), load_persona(persona))


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


def start(make_session: Callable[[str], Session], persona: str) -> tuple[Any, ...]:
    """Create this tab's session and show its starting tasks and settings."""
    session = make_session(persona)
    state = session.state
    pace = state.pace
    return (
        session,
        task_rows(state),
        [],
        f"{state.now:%Y-%m-%d %H:%M}",
        state.location,
        pace.daily_limit_min,
        pace.distribution,
        pace.hardest_first,
    )


def switch(make_session: Callable[[str], Session], persona: str) -> tuple[Any, ...]:
    """Replace this tab's session with a fresh one for another persona."""
    return (
        *start(make_session, persona),
        [],
        gr.update(visible=False),
        gr.update(interactive=True),
    )


def apply(  # noqa: PLR0913 - one argument per control
    clock: str,
    location: str,
    daily_limit: float,
    distribution: float,
    *,
    hardest_first: bool,
    session: Session | None,
) -> None:
    """Push the clock, location and pace controls into the session."""
    live = _live(session)
    try:
        now = datetime.fromisoformat(clock)
    except ValueError as exc:
        raise gr.Error(BAD_CLOCK) from exc
    pace = live.state.pace.model_copy(
        update={
            "daily_limit_min": int(daily_limit),
            "distribution": distribution,
            "hardest_first": hardest_first,
        }
    )
    try:
        live.update_session(now=now, location=location, pace=pace)
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc
    gr.Info("Settings updated.")


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


def build_app(make_session: Callable[[str], Session] = default_session) -> gr.Blocks:
    """Lay out the app; each browser tab gets its own session from `make_session(persona)`."""
    app = gr.Blocks(title="Planning delegate")
    places = [(p.name, p.id) for p in load_base().places]
    with app:
        session = gr.State()
        with gr.Row():
            persona = gr.Dropdown(list_personas(), value=PERSONA, label="Persona")
            clock = gr.Textbox(label="Clock", max_lines=1)
            location = gr.Dropdown(places, label="Location")
            daily_limit = gr.Number(label="Daily limit (min)", precision=0, minimum=0, maximum=1440)
            distribution = gr.Slider(0, 1, step=0.1, label="Spread (0 front-loaded, 1 even)")
            hardest_first = gr.Checkbox(label="Hardest first")
            apply_btn = gr.Button("Apply")
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
        settings = [clock, location, daily_limit, distribution, hardest_first]
        started = [session, tasks, log, *settings]
        app.load(lambda p: start(make_session, p), persona, started)
        persona.change(lambda p: switch(make_session, p), persona, [*started, chat, card, box])
        apply_btn.click(
            lambda c, loc, d, dist, h, s: apply(c, loc, d, dist, hardest_first=h, session=s),
            [*settings, session],
            None,
        )
        box.submit(send, [box, chat, session], shown)
        yes.click(lambda h, s: answer(approved=True, history=h, session=s), [chat, session], shown)
        no.click(lambda h, s: answer(approved=False, history=h, session=s), [chat, session], shown)
    return app
