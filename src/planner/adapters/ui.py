"""The Gradio app: chat, tasks and pace tabs, a header and a model settings sidebar.

`Controller` holds the handlers as plain methods over a per-tab `Tab`, so tests drive them without
a browser. `build_app` only lays components out and wires them. Exactly one handler, `chat`, is a
public API endpoint; every other handler is private, which also keeps the visitor's API key out of
the run history, since the key only ever reaches the private `connect` handler.
"""

from __future__ import annotations

import json
import logging
import random
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast, get_args

import gradio as gr
from pydantic import SecretStr

from planner.adapters.chat import History, Tab, new_tab, release_endpoint, stream_turn, use_endpoint
from planner.adapters.config import DUMMY_API_KEY, Endpoint
from planner.adapters.data import PERSONA, load_base, load_persona, load_rules
from planner.adapters.endpoint_guard import loggable
from planner.adapters.logs import emit
from planner.adapters.transcript import transcript
from planner.adapters.views import (
    PLACE_HEADERS,
    TASK_HEADERS,
    TRAVEL_HEADERS,
    examples,
    pace_chart,
    place_label,
    place_rows,
    task_rows,
    travel_rows,
)
from planner.app.prompt import forwarded
from planner.domain.models import Minutes, Pace, SessionState, Weekday

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import pandas as pd
    from gradio.components.chatbot import ExampleMessage

    from planner.adapters.backend import Backend
    from planner.adapters.config import Settings
    from planner.domain.models import BaseData

LOG = logging.getLogger("planner.ui")
START_PLACE: Final = "home"
BOX_ID: Final = "message-box"
# Marks a message as pasted: the page prepends it on submit and `chat` strips it.
PASTE_MARK: Final = "⁣"
# Ctrl/Cmd+V flags the next message as pasted; Ctrl+Shift+V (plain-text paste) does not.
WATCH_PASTE_JS: Final = f"""() => {{
  document.addEventListener("keydown", (e) => {{
    if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === "v"
        && e.target.closest("#{BOX_ID}")) window.plannerPasted = true;
  }}, true);
  document.addEventListener("input", (e) => {{
    if (e.target.closest("#{BOX_ID}") && !e.target.value) window.plannerPasted = false;
  }}, true);
}}"""
MARK_PASTE_JS: Final = f"""(text) => {{
  const pasted = window.plannerPasted && text.trim();
  window.plannerPasted = false;
  return pasted ? {json.dumps(PASTE_MARK)} + text : text;
}}"""
# Textbox has no built-in "disabled while empty": grey the submit arrow from the page.
ARROW_JS: Final = f"""(text) => {{
  const arrow = document.querySelector("#{BOX_ID} button[data-testid=submit-button]");
  if (arrow) arrow.disabled = !(text || "").trim();
}}"""
LEAVE_JS: Final = """() => {
  window.addEventListener("beforeunload", (e) => {
    if (window.plannerHasChat) { e.preventDefault(); e.returnValue = ""; }
  });
}"""
CHAT_CHANGED_JS: Final = "(chat) => { window.plannerHasChat = Boolean(chat && chat.length); }"
SESSION_LOST: Final = "This tab's session was lost, likely by a server restart. Reload the page."
KB: Final = 1000
MB: Final = 1000 * KB
MESSAGE_TOO_LONG: Final = (
    "Message is {actual:,} characters; the limit is {limit:,}. Shorten it or split it."
)
FILE_TOO_LARGE: Final = "File is {actual}; the limit is {limit}. Export a smaller session."


def size_text(size: int) -> str:
    """Show a byte count the way the limits are stated, such as "3.2 MB"."""
    return f"{size / MB:.1f} MB" if size >= MB else f"{size / KB:.1f} KB"


CHARGE_NOTICE: Final = (
    "Your own endpoint is called from this server with your key, which is held for this tab "
    "only: never stored, logged or exported. **Your provider will charge you for its use.**\n\n"
    "Only OpenAI-compatible chat endpoints are supported: a `/v1` base URL whose model supports "
    "tool calling."
)
THEME: Final = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(block_label_background_fill="transparent", block_title_background_fill="transparent")
CSS: Final = """
.compact table, .compact td, .compact th { font-size: 0.85rem !important; }
.chip { border: 1px solid var(--border-color-primary); border-radius: var(--radius-xl);
  padding: var(--spacing-sm) var(--spacing-xl); background: var(--background-fill-secondary);
  width: fit-content; }
.chip p { margin: 0; }
.header { flex-wrap: wrap; }
.header > div:first-child { flex: 0 0 auto !important; min-width: 15rem !important; }
.confirm-card { border: 2px solid var(--error-border-color) !important;
  border-radius: var(--radius-lg); padding: var(--spacing-md); }
"""
WEEKDAYS: Final = tuple(Weekday)
# The most a pace control accepts is the domain's own bound on a number of minutes.
MAX_MIN: Final[int] = next(c.le for c in get_args(Minutes)[1].metadata if hasattr(c, "le"))


def real_now() -> datetime:
    """The local time as a naive datetime, matching the session's naive clock."""
    return datetime.now().astimezone().replace(tzinfo=None)


def pace_from(  # noqa: PLR0913 - one argument per pace control
    daily: float | None,
    consecutive: float | None,
    long_break: float | None,
    distribution: float | None,
    *,
    hardest_first: bool,
    weekdays: list[float | None],
    current: Pace,
) -> Pace:
    """Build the pace the controls describe; an emptied control keeps its current value.

    Args:
        daily: Most work per day, in minutes.
        consecutive: Longest run of work before a long break; 0 turns the break off.
        long_break: Length of that break, in minutes.
        distribution: 0 for work front-loaded up to 1 for work spread evenly.
        hardest_first: Whether longer tasks without a deadline go first.
        weekdays: A limit per weekday, Monday first; None means the daily limit applies.
        current: The pace to fall back on for emptied controls.

    Returns:
        The new pace.
    """
    return Pace(
        daily_limit_min=current.daily_limit_min if daily is None else int(daily),
        max_consecutive_min=(
            current.max_consecutive_min if consecutive is None else int(consecutive) or None
        ),
        long_break_min=current.long_break_min if long_break is None else int(long_break),
        weekday_limit_min={
            d: int(v) for d, v in zip(WEEKDAYS, weekdays, strict=True) if v is not None
        },
        distribution=current.distribution if distribution is None else distribution,
        hardest_first=hardest_first,
    )


def pace_values(pace: Pace) -> tuple[Any, ...]:
    """The pace controls' values, in the order `pace_from` takes them."""
    return (
        pace.daily_limit_min,
        pace.max_consecutive_min or 0,
        pace.long_break_min,
        pace.distribution,
        pace.hardest_first,
        *(pace.weekday_limit_min.get(d) for d in WEEKDAYS),
    )


def _live(tab: Tab | None) -> Tab:
    # Gradio keeps the tab in server memory, so a restart or its time-to-live loses it.
    if tab is None:
        raise gr.Error(SESSION_LOST)
    return tab


def _write(prefix: str, content: str) -> str:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix=prefix, encoding="utf-8", delete=False
    ) as out:
        out.write(content)
    return out.name


class Controller:
    """The handlers of one app, sharing the backend, the base data and the clock."""

    def __init__(
        self,
        backend: Backend,
        settings: Settings,
        *,
        clock: Callable[[], datetime] = real_now,
        rng: random.Random | None = None,
    ) -> None:
        """Create the handlers.

        Args:
            backend: Supplies models and explains failures and cost.
            settings: The limits and defaults the handlers apply.
            clock: Tells the real time; replaced in tests.
            rng: Picks the example messages; replaced in tests.
        """
        self.backend = backend
        self.settings = settings
        self.clock = clock
        self.rng = rng or random.Random()  # noqa: S311 - example messages, nothing secret
        self.base: BaseData = load_base()
        self.rules = load_rules()

    # Views: what each handler returns for the page.

    def full(self, tab: Tab) -> tuple[Any, ...]:
        """Everything the page shows for a tab that has just begun: chat, tasks, pace, chart."""
        state = tab.session.state
        return (
            gr.Chatbot(
                value=[],
                examples=cast("list[ExampleMessage]", examples(state, self.base, self.rng)),
            ),
            gr.update(value="", interactive=True),
            task_rows(state),
            place_label(self.base, state.location),
            gr.update(visible=False),
            "",
            *pace_values(state.pace),
            pace_chart(state, self.base),
        )

    def refresh(self, tab: Tab | None) -> tuple[Any, ...]:
        """What changes after a turn: the box, the task table, the location and the card."""
        live = _live(tab)
        state = live.session.state
        asking = live.pending is not None
        return (
            gr.update(interactive=not asking),
            task_rows(state),
            place_label(self.base, state.location),
            gr.update(visible=asking),
            live.pending or "",
        )

    # Starting a tab.

    def boot(self) -> tuple[Any, ...]:
        """Start an empty tab on the real clock."""
        state = SessionState(now=self.clock(), location=START_PLACE)
        tab = new_tab(
            self.backend,
            self.base,
            self.rules,
            state,
            real_clock=True,
            max_rejections=self.settings.max_rejections,
        )
        return tab, *self.full(tab)

    def demo(self, tab: Tab | None) -> tuple[Any, ...]:
        """Start over from the sample persona on its simulated clock."""
        return self._restart(tab, load_persona(PERSONA))

    def import_state(self, tab: Tab | None, path: str | None) -> tuple[Any, ...]:
        """Start over from an exported state file."""
        file = Path(path or "")
        # Gradio's upload cap rejects most oversized files before this runs; this names the sizes.
        if file.is_file() and (size := file.stat().st_size) > self.settings.import_max_bytes:
            emit(
                LOG,
                "limit_hit",
                level=logging.WARNING,
                limit="import_max_bytes",
                actual=size,
                allowed=self.settings.import_max_bytes,
            )
            raise gr.Error(
                FILE_TOO_LARGE.format(
                    actual=size_text(size), limit=size_text(self.settings.import_max_bytes)
                )
            )
        try:
            state = SessionState.model_validate_json(file.read_text("utf-8"))
            if state.location not in self.base.place_ids:
                msg = f"unknown place: {state.location}"
                raise ValueError(msg)  # noqa: TRY301 - one message for every way a file is bad
        except (OSError, ValueError) as exc:
            emit(LOG, "import_failed", level=logging.WARNING, error_class=type(exc).__name__)
            raise gr.Error(f"Could not import that file: {exc}") from exc
        emit(LOG, "session_import", size_bytes=file.stat().st_size, tasks=len(state.tasks))
        return self._restart(tab, state)

    def expired(self, tab: Tab | None) -> None:
        """Log that Gradio dropped a tab's state: its time-to-live ran out or the tab went away."""
        if tab is not None:
            emit(LOG, "session_end", session=tab.session.log_id, turns=len(tab.turns))
            release_endpoint(tab.custom)

    def _restart(self, old: Tab | None, state: SessionState) -> tuple[Any, ...]:
        custom = old.custom if old else None
        try:
            tab = new_tab(
                self.backend,
                self.base,
                self.rules,
                state,
                real_clock=False,
                max_rejections=self.settings.max_rejections,
                custom=custom,
            )
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        return tab, *self.full(tab)

    # Settings.

    def connect(
        self, tab: Tab | None, choice: str, url: str, model: str, key: str
    ) -> tuple[str, str]:
        """Switch this tab to the default model or to the visitor's own endpoint.

        Returns:
            A status line, and an empty string to clear the key box.
        """
        live = _live(tab)
        custom = None
        try:
            if choice == "custom":
                if not (url.strip() and model.strip()):
                    msg = "Enter the endpoint's base URL and model name."
                    raise ValueError(msg)  # noqa: TRY301 - reported with the other bad inputs
                custom = Endpoint(
                    name="custom",
                    base_url=url.strip(),
                    model=model.strip(),
                    api_key=SecretStr(key or DUMMY_API_KEY),
                    timeout_s=self.settings.custom_timeout_s,
                )
            use_endpoint(live, self.backend, custom)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        if custom is None:
            emit(LOG, "endpoint_change", backend="default")
            return "Using the default model.", ""
        emit(
            LOG,
            "endpoint_change",
            backend="custom",
            model=custom.model,
            **loggable(custom.base_url),
        )
        return f"Using {custom.model} at {custom.base_url}.", ""

    def pace_changed(self, tab: Tab | None, *values: Any) -> pd.DataFrame:  # noqa: ANN401
        """Apply the pace controls to the session and return the updated chart."""
        live = _live(tab)
        daily, consecutive, long_break, distribution, hardest, *weekdays = values
        pace = pace_from(
            daily,
            consecutive,
            long_break,
            distribution,
            hardest_first=bool(hardest),
            weekdays=weekdays,
            current=live.session.state.pace,
        )
        try:
            live.session.update_session(now=self.clock() if live.real_clock else None, pace=pace)
        except RuntimeError as exc:
            gr.Warning(str(exc))
        return pace_chart(live.session.state, self.base)

    # Chat.

    def chat(
        self,
        text: str,
        history: History,
        tab: Tab | None,
        profile: gr.OAuthProfile | None = None,
    ) -> Iterator[tuple[Any, Any, Any]]:
        """Send a message to the agent and stream its reply and tool calls into the chat.

        A caller without a page (an API client) has no tab yet, so one is started on the real
        clock and kept for that client's session.

        A message the page marked as pasted goes to the model fenced as untrusted data. A blank
        message changes nothing. This is the app's only public API endpoint.
        """
        live = tab or self.boot()[0]
        pasted = text.startswith(PASTE_MARK)
        text = text.removeprefix(PASTE_MARK)
        if not text.strip():
            yield gr.skip(), gr.skip(), live
            return
        if len(text) > self.settings.message_max_chars:
            emit(
                LOG,
                "limit_hit",
                level=logging.WARNING,
                limit="message_max_chars",
                actual=len(text),
                allowed=self.settings.message_max_chars,
            )
            note = MESSAGE_TOO_LONG.format(actual=len(text), limit=self.settings.message_max_chars)
            yield [*history, {"role": "assistant", "content": note}], gr.skip(), live
            return
        if live.real_clock:
            live.session.update_session(now=self.clock())
        prompt = forwarded(text) if pasted else text
        shown = [*history, {"role": "user", "content": f"(forwarded)\n{text}" if pasted else text}]
        yield shown, gr.update(value="", interactive=False), live
        for chat in stream_turn(
            live,
            self.backend,
            shown,
            lambda: live.session.stream(prompt),
            said=text,
            signed_in=profile is not None,
        ):
            yield chat, gr.skip(), gr.skip()

    def approve(
        self, tab: Tab | None, history: History, profile: gr.OAuthProfile | None = None
    ) -> Iterator[History]:
        """Run the call the user said yes to."""
        return self._answer(tab, history, profile, approved=True)

    def decline(
        self, tab: Tab | None, history: History, profile: gr.OAuthProfile | None = None
    ) -> Iterator[History]:
        """Cancel the call the user said no to."""
        return self._answer(tab, history, profile, approved=False)

    def _answer(
        self, tab: Tab | None, history: History, profile: gr.OAuthProfile | None, *, approved: bool
    ) -> Iterator[History]:
        live = _live(tab)
        live.pending = None
        return stream_turn(
            live,
            self.backend,
            history,
            lambda: live.session.resume_stream(approved),
            said="(confirmed)" if approved else "(declined)",
            signed_in=profile is not None,
        )

    def pick_example(self, picked: gr.SelectData) -> str:
        """Put the clicked example into the message box."""
        value = picked.value
        return str(value["text"] if isinstance(value, dict) else value)

    # Files.

    def export_state(self, tab: Tab | None) -> Any:  # noqa: ANN401 - a component update
        """Write the session's state to a JSON file and offer it for download."""
        live = _live(tab)
        content = live.session.state.model_dump_json(indent=2)
        path = _write("session_", content)
        emit(LOG, "session_export", size_bytes=Path(path).stat().st_size)
        return gr.update(value=path, visible=True)

    def export_transcript(self, tab: Tab | None) -> Any:  # noqa: ANN401 - a component update
        """Write the transcript (chat, every call stage, usage, settings, tasks) for download."""
        live = _live(tab)
        record = transcript(
            live.session.state,
            live.turns,
            custom=live.custom,
            real_clock=live.real_clock,
            exported_at=self.clock(),
        )
        content = record.model_dump_json(indent=2)
        path = _write("transcript_", content)
        emit(
            LOG, "transcript_download", size_bytes=Path(path).stat().st_size, turns=len(live.turns)
        )
        return gr.update(value=path, visible=True)


def build_app(  # noqa: PLR0915 - the layout is one flat declaration of components and wiring
    backend: Backend, settings: Settings, *, controller: Controller | None = None
) -> gr.Blocks:
    """Lay out the app; each browser tab gets its own session, held server-side."""
    ctl = controller or Controller(backend, settings)
    app = gr.Blocks(title="Planning delegate")
    with app:
        tab = gr.State(time_to_live=settings.state_ttl_s, delete_callback=ctl.expired)
        with gr.Row(equal_height=True, elem_classes="header"):
            gr.Markdown("# Planning delegate", min_width=220)
            chip = gr.Markdown(elem_classes="chip")
            demo_btn = gr.Button("Try a demo", size="sm", min_width=100)
            export_btn = gr.Button("💾 Export", size="sm", min_width=90)
            import_btn = gr.UploadButton("📂 Import", file_types=[".json"], size="sm", min_width=90)
            transcript_btn = gr.Button("📝 Transcript", size="sm", min_width=100)
            settings_btn = gr.Button("⚙", size="sm", min_width=40)
        download = gr.File(label="Download", visible=False)
        with gr.Tabs():
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(show_label=False, height="60vh", layout="bubble")
                with gr.Group(visible=False, elem_classes="confirm-card") as card:
                    question = gr.Markdown()
                    with gr.Row():
                        yes = gr.Button("Yes, do it", variant="stop")
                        no = gr.Button("No, keep it")
                box = gr.Textbox(
                    elem_id=BOX_ID,
                    show_label=False,
                    placeholder="Ask me to plan something, or paste a message to look at",
                    max_lines=8,
                    max_length=settings.message_max_chars,
                    submit_btn=True,
                )
            with gr.Tab("Tasks"):
                tasks = gr.Dataframe(
                    headers=TASK_HEADERS,
                    label="Tasks",
                    interactive=False,
                    wrap=True,
                    elem_classes="compact",
                )
                with gr.Accordion("Places (read only)", open=False):
                    gr.Dataframe(
                        value=place_rows(ctl.base),
                        headers=PLACE_HEADERS,
                        label="Opening hours",
                        interactive=False,
                        wrap=True,
                        elem_classes="compact",
                    )
                    gr.Dataframe(
                        value=travel_rows(ctl.base),
                        headers=TRAVEL_HEADERS,
                        label="Travel",
                        interactive=False,
                        wrap=True,
                        elem_classes="compact",
                    )
            with gr.Tab("Pace"):
                with gr.Row():
                    daily = gr.Number(
                        label="Daily limit (min)", precision=0, minimum=0, maximum=MAX_MIN
                    )
                    consecutive = gr.Number(
                        label="Longest run before a long break (min)",
                        info="0 turns long breaks off",
                        precision=0,
                        minimum=0,
                        maximum=MAX_MIN,
                    )
                    long_break = gr.Number(
                        label="Long break (min)", precision=0, minimum=0, maximum=MAX_MIN
                    )
                distribution = gr.Slider(
                    0, 1, step=0.1, label="Spread", info="0 front-loaded, 1 even"
                )
                hardest = gr.Checkbox(
                    label="Hardest first", info="Among tasks without a deadline, longer first"
                )
                with (
                    gr.Accordion("Limit per weekday (min); empty uses the daily limit", open=False),
                    gr.Row(),
                ):
                    per_day = [
                        gr.Number(label=d.value.title(), precision=0, minimum=0, maximum=MAX_MIN)
                        for d in WEEKDAYS
                    ]
                chart = gr.BarPlot(
                    x="day",
                    y="minutes",
                    title="Planned minutes per day for a sample 10-hour task",
                    x_title="",
                    y_title="Minutes",
                )
        with gr.Sidebar(label="Model", open=False, position="right") as sidebar:
            gr.Markdown("### Model")
            choice = gr.Radio(
                [("Default model", "default"), ("My own endpoint", "custom")],
                value="default",
                label="Backend",
            )
            with gr.Group(visible=False) as own:
                url = gr.Textbox(
                    label="Base URL", placeholder="https://api.example.com/v1", max_lines=1
                )
                model = gr.Textbox(label="Model", max_lines=1)
                key = gr.Textbox(label="API key", type="password", max_lines=1)
                gr.Markdown(CHARGE_NOTICE)
            apply_btn = gr.Button("Use this backend", variant="primary", size="sm")
            status = gr.Markdown()
            gr.LoginButton()

        pace: list[Any] = [daily, consecutive, long_break, distribution, hardest, *per_day]
        reset = [tab, chatbot, box, tasks, chip, card, question, *pace, chart]
        after = [box, tasks, chip, card, question]
        sent = [box, chatbot, tab]

        app.load(ctl.boot, None, reset, api_visibility="private")
        app.load(None, None, None, js=WATCH_PASTE_JS)
        app.load(None, None, None, js=LEAVE_JS)
        app.load(None, box, None, js=ARROW_JS)
        box.change(None, box, None, js=ARROW_JS)
        chatbot.change(None, chatbot, None, js=CHAT_CHANGED_JS)
        demo_btn.click(ctl.demo, tab, reset, api_visibility="private")
        import_btn.upload(ctl.import_state, [tab, import_btn], reset, api_visibility="private")
        export_btn.click(ctl.export_state, tab, download, api_visibility="private")
        transcript_btn.click(ctl.export_transcript, tab, download, api_visibility="private")
        # Not queued: opening the sidebar must never wait behind, or be refused by, a full queue.
        settings_btn.click(
            lambda: gr.Sidebar(open=True), None, sidebar, api_visibility="private", queue=False
        )
        choice.change(
            lambda c: gr.update(visible=c == "custom"), choice, own, api_visibility="private"
        )
        apply_btn.click(
            ctl.connect, [tab, choice, url, model, key], [status, key], api_visibility="private"
        )
        gr.on(
            [c.input for c in pace],
            ctl.pace_changed,
            [tab, *pace],
            chart,
            api_visibility="private",
            show_progress="hidden",
        )

        box.submit(None, box, box, js=MARK_PASTE_JS).then(
            ctl.chat,
            sent,
            [chatbot, box, tab],
            api_name="chat",
            concurrency_limit=settings.concurrency_limit,
        ).then(ctl.refresh, tab, after, api_visibility="private")
        chatbot.example_select(ctl.pick_example, None, box, api_visibility="private").then(
            ctl.chat,
            sent,
            [chatbot, box, tab],
            api_visibility="private",
            concurrency_limit=settings.concurrency_limit,
        ).then(ctl.refresh, tab, after, api_visibility="private")
        for button, handler in ((yes, ctl.approve), (no, ctl.decline)):
            button.click(handler, [tab, chatbot], chatbot, api_visibility="private").then(
                ctl.refresh, tab, after, api_visibility="private"
            )
    app.queue(max_size=settings.queue_max_size)
    # The login button registers its own load check as a public endpoint; hide it.
    for fn in app.fns.values():
        if fn.name == "_check_login_status":
            fn.api_visibility = "private"
    return app
