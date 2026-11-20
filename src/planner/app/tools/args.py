# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
"""Argument models for the eight tools.

Single source of truth: each model yields the JSON schema shown to the LLM and performs the
shape checks (types, ranges, no unexpected arguments). State-dependent checks (IDs, places,
horizon, cycles) live in `planner.app.validation`. Docstrings are model-facing descriptions.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, ClassVar, Final, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, NaiveDatetime, model_validator
from pydantic_core import PydanticCustomError

from planner.domain.models import Mode, TaskStatus, TaskType

MAX_TASK_MIN: Final = 720
"""The longest one task can be, in minutes (12 hours)."""
Duration = Annotated[int, Field(strict=True, ge=5, le=MAX_TASK_MIN)]
Text = Annotated[str, Field(strict=True, min_length=1)]
DurationQuote = Annotated[
    str | None,
    Field(
        strict=True,
        description="Required with duration_min: the user's own words that state it, copied "
        "exactly, like 'about an hour'. If the user gave no duration, omit both.",
    ),
]
DeadlineQuote = Annotated[
    str | None,
    Field(
        strict=True,
        description="Required with deadline: the user's own words that state it, copied "
        "exactly from their message. If the user gave no deadline, omit both.",
    ),
]
QUOTE_FIELDS: Final = frozenset({"duration_quote", "deadline_quote"})
"""Evidence for a stated value: checked by validation, never stored on a task."""
END_OF_DAY: Final = dt.time(23, 59)
"""The time a date-only deadline is read as."""


def _drop_z(value: object) -> object:
    """Read a trailing "Z" as the session's own clock: the model adds it out of habit."""
    if isinstance(value, str) and value[-1:] in {"Z", "z"}:
        return value[:-1]
    return value


def _require_time(value: object) -> object:
    """Reject date-only strings: pydantic would read them as midnight, silently wrong."""
    value = _drop_z(value)
    if isinstance(value, str) and "T" not in value:
        msg = "must include a time, like 2026-10-28T17:15"
        raise PydanticCustomError("bad_datetime", msg)
    return value


def _end_of_day(value: object) -> object:
    """Read a date-only deadline as the end of that day, not its midnight start."""
    value = _drop_z(value)
    if isinstance(value, str) and "T" not in value:
        try:
            day = dt.date.fromisoformat(value)
        except ValueError:
            return value  # let the datetime parser report it
        return dt.datetime.combine(day, END_OF_DAY)
    return value


TimeOfDay = Annotated[NaiveDatetime, BeforeValidator(_require_time)]
Deadline = Annotated[NaiveDatetime, BeforeValidator(_end_of_day)]


class ToolArgs(BaseModel):
    """Base for every tool's arguments: immutable, and unexpected arguments are an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: ClassVar[str]


class AddTaskArgs(ToolArgs):
    """Capture a task. Omit anything the user has not said; never guess a duration or place.

    A duration or deadline needs its quote: the user's words that state it.
    """

    tool_name: ClassVar[str] = "add_task"

    title: Text
    type: TaskType | None = None
    duration_min: Duration | None = None
    duration_quote: DurationQuote = None
    deadline: Deadline | None = None
    deadline_quote: DeadlineQuote = None
    place: Text | None = None
    earliest_start: TimeOfDay | None = None
    prerequisites: tuple[Text, ...] = ()


class TaskPatch(BaseModel):
    """Fields to change. Omit the ones to keep."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: Text | None = None
    type: TaskType | None = None
    duration_min: Duration | None = None
    duration_quote: DurationQuote = None
    deadline: Deadline | None = None
    deadline_quote: DeadlineQuote = None
    place: Text | None = None
    earliest_start: TimeOfDay | None = None
    prerequisites: tuple[Text, ...] | None = None
    status: TaskStatus | None = None

    @model_validator(mode="after")
    def _not_empty(self) -> Self:
        if not self.changes():
            raise PydanticCustomError("bad_value", "at least one field must be given")
        return self

    def changes(self) -> dict[str, object]:
        """The task fields to change, without the quotes that back them."""
        return self.model_dump(exclude_none=True, exclude=set(QUOTE_FIELDS))


class UpdateTaskArgs(ToolArgs):
    """Edit a task, or mark it done with fields {"status": "done"}."""

    tool_name: ClassVar[str] = "update_task"

    task_id: Text
    fields: TaskPatch


class DeleteTaskArgs(ToolArgs):
    """Delete one task by id, or every task with "all". Always needs user confirmation."""

    tool_name: ClassVar[str] = "delete_task"

    task_id: Text


class TravelTimeArgs(ToolArgs):
    """Look up travel minutes between two known places. Returns unknown if the route is missing."""

    tool_name: ClassVar[str] = "travel_time"

    from_place: Text
    to_place: Text
    mode: Mode


class CheckFeasibilityArgs(ToolArgs):
    """Check whether a task can start at a given time: window open, prerequisites, travel."""

    tool_name: ClassVar[str] = "check_feasibility"

    task_id: Text
    start: TimeOfDay
    from_place: Text


class RecommendNowArgs(ToolArgs):
    """Recommend what to do now, using the stored time, location and pace. Takes no arguments."""

    tool_name: ClassVar[str] = "recommend_now"


class SetLocationArgs(ToolArgs):
    """Record where the user is now, when they say they have moved to a known place."""

    tool_name: ClassVar[str] = "set_location"

    place_id: Text


class PlanDayArgs(ToolArgs):
    """Plan one day: prerequisites first, least slack first, travel buffers, pace, daily limit."""

    tool_name: ClassVar[str] = "plan_day"

    date: dt.date


_ALL_TOOLS: Final[tuple[type[ToolArgs], ...]] = (
    AddTaskArgs,
    UpdateTaskArgs,
    DeleteTaskArgs,
    TravelTimeArgs,
    CheckFeasibilityArgs,
    RecommendNowArgs,
    PlanDayArgs,
    SetLocationArgs,
)
TOOLS_BY_NAME: Final[dict[str, type[ToolArgs]]] = {t.tool_name: t for t in _ALL_TOOLS}
