"""What the UI shows, built from the session and the base data: tables, chat rows, examples, chart.

Pure functions with no Gradio state, so tests check them directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Final

import pandas as pd
import plotly.graph_objects as go

from planner.app import scheduler
from planner.domain.models import Mode, Task, TaskStatus, TaskType, Weekday

if TYPE_CHECKING:
    from collections.abc import Sequence
    from random import Random

    from planner.adapters.session import Usage
    from planner.app.calls import CallRecord
    from planner.domain.models import BaseData, Place, SessionState

type Rows = list[list[str]]
type Message = dict[str, Any]

TASK_HEADERS: Final = ["ID", "Title", "Type", "Min", "Place", "Deadline", "Status"]
PLACE_HEADERS: Final = ["Place", "ID", "Kind", "Open"]
TRAVEL_HEADERS: Final = ["From", "To", "Mode", "Min", "Buffer"]
SAMPLE_MIN: Final = 600
SAMPLE_DAYS: Final = 4
"""The Pace chart plans a SAMPLE_MIN-minute flexible task due SAMPLE_DAYS days after today."""
EXAMPLES_SHOWN: Final = 3
FILL_ALPHA: Final = 0.3
EDGE_ALPHA: Final = 0.35
"""How opaque the edges of the Places graph are."""
"""How opaque the area under the Pace curve is."""
THINKING_WORDS: Final = (
    "Thinking…",
    "Planning…",
    "Pondering…",
    "Crunching…",
    "Buzzing…",
    "Weighing options…",
    "Checking the clock…",
    "Reading your tasks…",
    "Working it out…",
    "Almost there…",
)
"""The texts the pending line rotates through while the agent is silent."""
_REFUSED: Final = ("rejected", "declined")


def task_rows(state: SessionState) -> Rows:
    """Describe each task as a table row."""
    return [
        [
            t.id,
            t.title,
            t.type.value.replace("_", " "),
            "" if t.duration_min is None else str(t.duration_min),
            t.place,
            f"{t.deadline:%a %d/%m %H:%M}" if t.deadline else "",
            t.status.value,
        ]
        for t in state.tasks
    ]


def place_rows(base: BaseData) -> Rows:
    """Describe each known place and its opening hours as a table row."""
    return [[p.name, p.id, p.category, _hours(p)] for p in base.places]


def travel_rows(base: BaseData) -> Rows:
    """Describe each leg of the travel table as a table row."""
    return [
        [leg.origin, leg.destination, leg.mode.value, str(leg.minutes), str(leg.buffer_min)]
        for leg in base.travel
    ]


def place_label(base: BaseData, place_id: str) -> str:
    """Name a place for the location chip, such as `📍 Office`."""
    place = base.place(place_id)
    return f"📍 {place.name if place else place_id}"


def call_message(records: Sequence[CallRecord], *, waiting: bool = False) -> Message:
    """Show one tool call as a single row that grows as the call passes each stage.

    Args:
        records: Every stage the call has passed so far, oldest first.
        waiting: Whether the call is parked on the user's yes or no.

    Returns:
        A chat row: a pending spinner until the call ends, a badge for its state in the title,
        and one line per stage in the expandable body.
    """
    last = records[-1]
    state, final = _call_state(last, waiting=waiting)
    title = f'{last.tool} <span class="badge {state.css}">{state.text}</span>'
    metadata: dict[str, str] = {"title": title, "status": "done" if final else "pending"}
    lines = "\n".join(f"- {r.stage}: {r.outcome}" for r in records)
    content = f"`{last.tool}` `{last.args}`\n\n{lines}"
    return {"role": "assistant", "content": content, "metadata": metadata}


@dataclass(frozen=True)
class _CallState:
    text: str
    css: str


def _call_state(last: CallRecord, *, waiting: bool) -> tuple[_CallState, bool]:
    """Name the state a call is in after `last`, and say whether the call is over."""
    if last.stage == "act":
        return _CallState("done", "ok"), True
    if last.outcome.startswith(_REFUSED):
        word = "declined" if last.outcome.startswith("declined") else "rejected"
        return _CallState(word, "bad"), True
    if waiting:
        return _CallState("waiting for you", "wait"), False
    if last.stage == "check":
        return _CallState("checked", "run"), False
    return _CallState("running", "run"), False


def thinking(tick: int, tool: str | None = None) -> Message:
    """The pending line shown while the agent works: a tool hint, else a rotating word.

    It is plain reply text with a small spinner, not a collapsible row, so it never nests a box
    inside the bubble.
    """
    text = f"Running {tool}…" if tool else THINKING_WORDS[tick % len(THINKING_WORDS)]
    return {"role": "assistant", "content": f'<span class="spinner"></span>{text}'}


def usage_line(seconds: float, usage: Usage | None) -> str:
    """Summarise a turn, such as `3.2 s · 1,200 in / 40 out tokens`."""
    parts = [f"{seconds:.1f} s"]
    if usage:
        parts.append(f"{usage.input_tokens:,} in / {usage.output_tokens:,} out tokens")
    return " · ".join(parts)


def examples(state: SessionState, base: BaseData, rng: Random) -> list[dict[str, str]]:
    """Pick a few example messages that make sense for this session's tasks and places."""
    here = base.place(state.location)
    near = [p for p in base.places if p.id != state.location and _drivable(base, state, p)]
    tomorrow = state.now + timedelta(days=1)
    later = state.now + timedelta(days=3)
    pool = ["What should I do now?", f"Plan my {tomorrow:%A}"]
    if here and near:
        other = rng.choice(near)
        pool += [
            f"How long is the drive from {here.name} to {other.name}?",
            f"I'm at {other.name} now",
        ]
    todo = [t for t in state.tasks if t.status is TaskStatus.TODO]
    if todo:
        task = rng.choice(todo)
        pool += [f"Mark {task.id} as done", f"Can I start {task.id} on {later:%A} at 10:00?"]
    else:
        pool += [
            f"Add a 2-hour report due {later:%A} 17:00",
            f"Add: buy groceries, 45 minutes, before {tomorrow:%A} 18:00",
        ]
    return [{"text": text} for text in rng.sample(pool, EXAMPLES_SHOWN)]


def pace_chart(state: SessionState, base: BaseData) -> pd.DataFrame:
    """Plan minutes per day, from today to its deadline, for a sample task under this pace.

    The domain scheduler plans each day for a session holding only what is left of the sample
    task, so the chart shows what `plan_day` would do with these pace settings.
    """
    today = state.now.date()
    days = [today + timedelta(days=i) for i in range(SAMPLE_DAYS + 1)]
    deadline = datetime.combine(days[-1], time(23, 59))
    left = SAMPLE_MIN
    minutes: list[int] = []
    for day in days:
        planned = 0
        if left:
            sample = Task(
                id="sample",
                title="Sample task",
                type=TaskType.FLEXIBLE,
                duration_min=left,
                deadline=deadline,
            )
            plan = scheduler.plan_day(state.with_tasks((sample,)), base, day)
            planned = sum((b.end - b.start) // timedelta(minutes=1) for b in plan.blocks)
        minutes.append(planned)
        left -= min(planned, left)
    return pd.DataFrame({"day": pd.to_datetime(days), "minutes": minutes})


def pace_figure(state: SessionState, base: BaseData, color: str) -> go.Figure:
    """The pace chart as a smooth line over a translucent area, one tick per day in date order.

    Args:
        state: The session whose pace and clock the chart shows.
        base: Places and travel times.
        color: The line colour as `#rrggbb`; the app passes its theme's primary colour.
    """
    chart = pace_chart(state, base)
    red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    figure = go.Figure(
        go.Scatter(
            x=chart["day"],
            y=chart["minutes"],
            mode="lines+markers",
            line={"shape": "spline", "color": color},
            fill="tozeroy",
            fillcolor=f"rgba({red}, {green}, {blue}, {FILL_ALPHA})",
            hovertemplate="%{x|%a %d %b}: %{y} min<extra></extra>",
        )
    )
    figure.update_layout(
        title="Planned minutes per day for a sample 10-hour task",
        margin={"l": 50, "r": 50, "t": 50, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"tickmode": "array", "tickvals": list(chart["day"]), "tickformat": "%a %d %b"},
        yaxis={"title": "Minutes", "rangemode": "tozero"},
    )
    return figure


def places_figure(base: BaseData, color: str) -> go.Figure:
    """The places as a read-only graph: a node per place, an edge per pair with its minutes.

    Places sit on a circle (a fixed layout, so the picture never shifts). An edge is labelled with
    the minutes of each mode between its two places; hovering a label names the modes.

    Args:
        base: Places and the travel table.
        color: The colour as `#rrggbb`; the app passes its theme's primary colour.
    """
    angle = {p.id: 2 * math.pi * i / len(base.places) for i, p in enumerate(base.places)}
    at = {pid: (math.cos(a), math.sin(a)) for pid, a in angle.items()}
    legs: dict[tuple[str, str], list[str]] = {}
    for leg in base.travel:
        pair = (min(leg.origin, leg.destination), max(leg.origin, leg.destination))
        text = f"{leg.mode.value} {leg.minutes}"
        if text not in legs.setdefault(pair, []):
            legs[pair].append(text)
    lines_x: list[float | None] = []
    lines_y: list[float | None] = []
    for a, b in legs:
        lines_x += [at[a][0], at[b][0], None]
        lines_y += [at[a][1], at[b][1], None]
    middles = [((at[a][0] + at[b][0]) / 2, (at[a][1] + at[b][1]) / 2) for a, b in legs]
    labels = [" / ".join(t.rsplit(" ", 1)[1] for t in texts) for texts in legs.values()]
    figure = go.Figure(
        [
            go.Scatter(
                x=lines_x,
                y=lines_y,
                mode="lines",
                line={"color": color, "width": 1},
                opacity=EDGE_ALPHA,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=[m[0] for m in middles],
                y=[m[1] for m in middles],
                mode="text",
                text=labels,
                textfont={"size": 10},
                hovertext=[", ".join(f"{t} min" for t in texts) for texts in legs.values()],
                hoverinfo="text",
            ),
            go.Scatter(
                x=[at[p.id][0] for p in base.places],
                y=[at[p.id][1] for p in base.places],
                mode="markers+text",
                text=[p.name for p in base.places],
                textposition="top center",
                marker={"size": 14, "color": color},
                hoverinfo="skip",
            ),
        ]
    )
    axis = {"visible": False, "range": [-1.4, 1.4]}
    figure.update_layout(
        showlegend=False,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=axis,
        yaxis={**axis, "scaleanchor": "x"},
        height=480,
    )
    return figure


def _drivable(base: BaseData, state: SessionState, place: Place) -> bool:
    return base.leg(state.location, place.id, Mode.DRIVE) is not None


def _hours(place: Place) -> str:
    if not place.hours:
        return "always"
    order = list(Weekday)
    return "; ".join(
        f"{' '.join(d.value for d in sorted(w.days, key=order.index))} "
        f"{w.opens:%H:%M}-{w.closes:%H:%M}"
        for w in place.hours
    )
