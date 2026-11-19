# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
"""What the UI shows, built from the session and the base data: tables, chat rows, examples, chart.

Pure functions with no Gradio state, so tests check them directly.
"""

from __future__ import annotations

import html
import itertools
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
"""How opaque the area under the Pace curve is."""
EDGE_ALPHA: Final = 0.35
HOVER_LABEL: Final = {
    "bgcolor": "#ffffff",
    "bordercolor": "#e5e7eb",
    "font": {
        "family": "Source Sans Pro, ui-sans-serif, system-ui, sans-serif",
        "size": 13,
        "color": "#1f2937",
    },
    "align": "left",
}
"""The Places graph's hover box: neutral and light, in the theme's font, not trace-coloured."""
"""How opaque the edges of the Places graph are."""
NODE_SIZE: Final = 32
"""The diameter of a place on the Places graph, in pixels."""
CURVE_STEPS: Final = 12
"""Points drawn between two days on the Pace curve."""
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


def usage_footer(seconds: float, usage: Usage | None) -> str:
    """The turn's usage as the footer of a reply, for the page to show under the copy icon.

    Gradio 6.31 has no per-message footer, so the reply carries this span and a few lines of
    CSS (`.turn-usage` in `ui.CSS`) place it under the message's copy button.
    """
    return f'<span class="turn-usage">{usage_line(seconds, usage)}</span>'


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
    days = list(chart["day"])
    minutes = [int(m) for m in chart["minutes"]]
    curve = monotone_curve(minutes, CURVE_STEPS)
    figure = go.Figure(
        [
            go.Scatter(
                x=[days[0] + timedelta(days=t) for t, _ in curve],
                y=[v for _, v in curve],
                mode="lines",
                line={"color": color},
                fill="tozeroy",
                fillcolor=f"rgba({red}, {green}, {blue}, {FILL_ALPHA})",
                hoverinfo="skip",
            ),
            go.Scatter(
                x=days,
                y=minutes,
                mode="markers",
                marker={"color": color},
                hovertemplate="%{x|%a %d %b}: %{y} min<extra></extra>",
            ),
        ]
    )
    figure.update_layout(
        showlegend=False,
        margin={"l": 50, "r": 50, "t": 20, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "tickmode": "array",
            "tickvals": days,
            "tickformat": "%a %d %b",
            "range": [days[0] - timedelta(hours=18), days[-1] + timedelta(hours=18)],
            "ticklabelstandoff": 8,
        },
        yaxis={"title": "Minutes", "rangemode": "tozero"},
    )
    return figure


def monotone_curve(values: Sequence[int], steps: int) -> list[tuple[float, float]]:
    """Draw a smooth curve through one value per day that never overshoots its neighbours.

    A cubic Hermite spline whose slopes follow Fritsch and Carlson: a day that is a peak, a
    trough or flat has slope 0, so the curve stays between each pair of neighbouring values
    (a plain spline dips below a flat stretch before rising). Plotly's own spline cannot give
    that guarantee.

    Args:
        values: One value per day; their positions are 0, 1, 2, and so on.
        steps: Points to draw in each interval.

    Returns:
        (position, value) pairs from the first day to the last.
    """
    n = len(values)
    if n < 2:  # noqa: PLR2004 - a curve needs two points
        return [(float(i), float(v)) for i, v in enumerate(values)]
    delta = [values[i + 1] - values[i] for i in range(n - 1)]
    slope = [float(delta[0])]
    for before, after in itertools.pairwise(delta):
        slope.append(0.0 if before * after <= 0 else 2 * before * after / (before + after))
    slope.append(float(delta[-1]))
    curve: list[tuple[float, float]] = []
    for i in range(n - 1):
        for k in range(steps):
            t = k / steps
            h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
            h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
            value = h00 * values[i] + h10 * slope[i] + h01 * values[i + 1] + h11 * slope[i + 1]
            curve.append((i + t, value))
    curve.append((float(n - 1), float(values[-1])))
    return curve


def places_figure(base: BaseData, color: str) -> go.Figure:
    """The places as a read-only, interactive graph: a node per place, an edge per pair.

    Places sit on a circle (a fixed layout, so the picture never shifts). Hovering a node lists
    its edges with their travel minutes per mode; hovering the middle of an edge names its two
    places. The graph pans and zooms.

    Args:
        base: Places and the travel table.
        color: The colour as `#rrggbb`; the app passes its theme's primary colour.
    """
    angle = {p.id: 2 * math.pi * i / len(base.places) for i, p in enumerate(base.places)}
    at = {pid: (math.cos(a), math.sin(a)) for pid, a in angle.items()}
    name = {p.id: p.name for p in base.places}
    legs: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for leg in base.travel:
        pair = (min(leg.origin, leg.destination), max(leg.origin, leg.destination))
        option = (leg.mode.value, leg.minutes)
        if option not in legs.setdefault(pair, []):
            legs[pair].append(option)
    lines_x: list[float | None] = []
    lines_y: list[float | None] = []
    for a, b in legs:
        lines_x += [at[a][0], at[b][0], None]
        lines_y += [at[a][1], at[b][1], None]
    middles = [((at[a][0] + at[b][0]) / 2, (at[a][1] + at[b][1]) / 2) for a, b in legs]
    edge_hover = [
        _hover(f"{name[a]} to {name[b]}", [f"{m} {n} min" for m, n in options])
        for (a, b), options in legs.items()
    ]
    node_hover = [
        _hover(
            p.name,
            [
                f"{name[b if a == p.id else a]}: " + ", ".join(f"{m} {n} min" for m, n in options)
                for (a, b), options in legs.items()
                if p.id in (a, b)
            ],
        )
        for p in base.places
    ]
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
                mode="markers+text",
                text=edge_labels(base),
                textposition="top center",
                textfont={"size": 11},
                marker={"size": 16, "color": color, "opacity": 0},
                hovertemplate=edge_hover,
                hoverlabel=HOVER_LABEL,
            ),
            go.Scatter(
                x=[at[p.id][0] for p in base.places],
                y=[at[p.id][1] for p in base.places],
                mode="markers+text",
                text=[p.name for p in base.places],
                textposition="top center",
                textfont={"size": 14},
                marker={"size": NODE_SIZE, "color": color},
                hovertemplate=node_hover,
                hoverlabel=HOVER_LABEL,
            ),
        ]
    )
    axis = {"visible": False, "range": [-1.4, 1.4]}
    figure.update_layout(
        showlegend=False,
        dragmode="pan",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=axis,
        yaxis={**axis, "scaleanchor": "x"},
        height=520,
    )
    return figure


def _hover(title: str, lines: Sequence[str]) -> str:
    """A hover box body: a bold title, one line per item and no trace name."""
    return (
        "<br>".join([f"<b>{html.escape(title)}</b>", *map(html.escape, lines)]) + "<extra></extra>"
    )


def edge_labels(base: BaseData) -> list[str]:
    """The always-visible label for each edge of the Places graph: its fastest travel time."""
    fastest: dict[tuple[str, str], int] = {}
    for leg in base.travel:
        pair = (min(leg.origin, leg.destination), max(leg.origin, leg.destination))
        fastest[pair] = min(leg.minutes, fastest.get(pair, leg.minutes))
    return [f"{minutes} min" for minutes in fastest.values()]


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
