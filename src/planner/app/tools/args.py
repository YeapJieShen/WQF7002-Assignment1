"""Argument models for the seven tools.

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

Duration = Annotated[int, Field(strict=True, ge=5, le=720)]
Text = Annotated[str, Field(strict=True, min_length=1)]


def _require_time(value: object) -> object:
    """Reject date-only strings: pydantic would read them as midnight, silently wrong."""
    if isinstance(value, str) and "T" not in value:
        msg = "must include a time, like 2026-10-28T17:15"
        raise PydanticCustomError("bad_datetime", msg)
    return value


TimeOfDay = Annotated[NaiveDatetime, BeforeValidator(_require_time)]


class ToolArgs(BaseModel):
    """Base for every tool's arguments: immutable, and unexpected arguments are an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: ClassVar[str]


class AddTaskArgs(ToolArgs):
    """Capture a task. Omit anything the user has not said; never guess a duration or place."""

    tool_name: ClassVar[str] = "add_task"

    title: Text
    type: TaskType | None = None
    duration_min: Duration | None = None
    deadline: TimeOfDay | None = None
    place: Text | None = None
    earliest_start: TimeOfDay | None = None
    prerequisites: tuple[Text, ...] = ()


class TaskPatch(BaseModel):
    """Fields to change. Omit the ones to keep."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: Text | None = None
    type: TaskType | None = None
    duration_min: Duration | None = None
    deadline: TimeOfDay | None = None
    place: Text | None = None
    earliest_start: TimeOfDay | None = None
    prerequisites: tuple[Text, ...] | None = None
    status: TaskStatus | None = None

    @model_validator(mode="after")
    def _not_empty(self) -> Self:
        if not self.model_dump(exclude_none=True):
            raise PydanticCustomError("bad_value", "at least one field must be given")
        return self


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
)
TOOLS_BY_NAME: Final[dict[str, type[ToolArgs]]] = {t.tool_name: t for t in _ALL_TOOLS}
