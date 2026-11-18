"""What the UI shows, built from the session and the base data: tables, chat rows, examples, chart.

Pure functions with no Gradio state, so tests check them directly.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Final

import pandas as pd

from planner.app import scheduler
from planner.domain.models import Mode, Task, TaskStatus, TaskType, Weekday

if TYPE_CHECKING:
    from random import Random

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


def stage_message(record: CallRecord) -> Message:
    """Show one stage of a tool call as a collapsed row in the chat.

    The title says the tool, the stage and its verdict; a rejection's reason shows beside it.
    """
    verdict, _, reason = record.outcome.partition(": ")
    refused = record.outcome.startswith(_REFUSED)
    if record.stage == "act":
        verdict, reason = "done", ""
    icon = "❌" if refused else "✅"
    metadata: dict[str, str] = {"title": f"{icon} {record.tool} · {record.stage}: {verdict}"}
    if reason:
        metadata["log"] = reason
    metadata["status"] = "done"
    content = f"`{record.tool}` `{record.args}`\n\n```text\n{record.outcome}\n```"
    return {"role": "assistant", "content": content, "metadata": metadata}


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
    return pd.DataFrame({"day": [f"{d:%a %d %b}" for d in days], "minutes": minutes})


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
