# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
"""What each tool does once its call has been validated.

Handlers trust their arguments: ids, places and dates were checked by `planner.app.validation`.
They never edit the state; a change returns a new one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from planner.app import feasibility, scheduler
from planner.app.validation import DELETE_ALL
from planner.domain.models import ANYWHERE, SessionState, Task, TaskType

if TYPE_CHECKING:
    from planner.app.scheduler import DayPlan, Unscheduled
    from planner.app.tools.args import (
        AddTaskArgs,
        CheckFeasibilityArgs,
        DeleteTaskArgs,
        PlanDayArgs,
        SetLocationArgs,
        TravelTimeArgs,
        UpdateTaskArgs,
    )
    from planner.domain.models import BaseData

_GENERATED_ID = re.compile(r"T(\d+)")


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a tool hands back.

    Attributes:
        state: The session state after the tool ran; the same object if nothing changed.
        message: What happened, phrased for the model to relay to the user.
    """

    state: SessionState
    message: str


def add_task(state: SessionState, args: AddTaskArgs) -> ToolResult:
    """Add a task with a fresh id. Unstated type is flexible and unstated place is anywhere."""
    number = _next_number(state)
    task = Task(
        id=f"T{number}",
        title=args.title,
        type=args.type or TaskType.FLEXIBLE,
        duration_min=args.duration_min,
        deadline=args.deadline,
        place=args.place or ANYWHERE,
        earliest_start=args.earliest_start,
        prerequisites=args.prerequisites,
    )
    note = "" if task.duration_min else " It has no duration yet; ask the user for one."
    return ToolResult(
        state.with_tasks((*state.tasks, task), last_task_number=number),
        f"Added {task.id}: {task.title}.{note}",
    )


def update_task(state: SessionState, args: UpdateTaskArgs) -> ToolResult:
    """Change the given fields of a task."""
    changes = args.fields.model_dump(exclude_none=True)
    tasks = tuple(
        Task.model_validate({**t.model_dump(), **changes}) if t.id == args.task_id else t
        for t in state.tasks
    )
    return ToolResult(state.with_tasks(tasks), f"Updated {args.task_id}: {', '.join(changes)}.")


def delete_task(state: SessionState, args: DeleteTaskArgs) -> ToolResult:
    """Delete one task, or all of them. Confirmation is the caller's job, not this function's."""
    used = _highest_number(state)
    if args.task_id == DELETE_ALL:
        return ToolResult(
            state.with_tasks((), last_task_number=used), f"Deleted all {len(state.tasks)} tasks."
        )
    remaining = tuple(
        t.model_copy(
            update={"prerequisites": tuple(p for p in t.prerequisites if p != args.task_id)}
        )
        for t in state.tasks
        if t.id != args.task_id
    )
    return ToolResult(
        state.with_tasks(remaining, last_task_number=used), f"Deleted {args.task_id}."
    )


def travel_time(state: SessionState, base: BaseData, args: TravelTimeArgs) -> ToolResult:
    """Look up travel time from the table. A missing route is reported, not estimated."""
    leg = base.leg(args.from_place, args.to_place, args.mode)
    if leg is None:
        message = (
            f"No {args.mode} route from {args.from_place} to {args.to_place} in the travel table."
        )
    else:
        message = (
            f"{args.from_place} to {args.to_place} by {args.mode}: {leg.minutes} min "
            f"plus {leg.buffer_min} min buffer."
        )
    return ToolResult(state, message)


def check_feasibility(
    state: SessionState, base: BaseData, args: CheckFeasibilityArgs
) -> ToolResult:
    """Answer yes or no with the reasons; suggesting alternatives is left to the model."""
    task = _existing(state, args.task_id)
    result = feasibility.check_feasibility(
        state, base, task, start=args.start, from_place=args.from_place
    )
    if result.feasible and result.arrive and result.finish:
        message = (
            f"{task.id} fits: arrive {result.arrive:%a %H:%M}, finish {result.finish:%a %H:%M}."
        )
    else:
        message = f"{task.id} does not fit: {'; '.join(result.reasons)}."
    return ToolResult(state, message)


def plan_day(state: SessionState, base: BaseData, args: PlanDayArgs) -> ToolResult:
    """Plan one day and describe the blocks, the total work and what was left out."""
    plan = scheduler.plan_day(state, base, args.date)
    lines = [f"Plan for {args.date:%a %Y-%m-%d}:"]
    lines += [
        f"{b.start:%H:%M}-{b.end:%H:%M} {b.task_id} {_existing(state, b.task_id).title}"
        for b in plan.blocks
    ]
    if not plan.blocks:
        lines.append("Nothing scheduled.")
    lines.append(f"Planned work: {_work_minutes(state, plan)} min.")
    lines += _unscheduled_lines(state, plan.unscheduled, "Not scheduled:")
    return ToolResult(state, "\n".join(lines))


def recommend_now(state: SessionState, base: BaseData) -> ToolResult:
    """Recommend the most urgent task that can be started right now."""
    rec = scheduler.recommend_now(state, base)
    if rec.task_id is None or rec.arrive is None or rec.finish is None:
        lines = ["Nothing can be done right now."]
    else:
        task = _existing(state, rec.task_id)
        lines = [
            f"Do {task.id} {task.title} now: arrive {rec.arrive:%H:%M}, finish {rec.finish:%H:%M}."
        ]
    lines += _unscheduled_lines(state, rec.skipped, "Ruled out:")
    return ToolResult(state, "\n".join(lines))


def set_location(state: SessionState, base: BaseData, args: SetLocationArgs) -> ToolResult:
    """Move the user to a known place; travel from then on is counted from there."""
    place = base.place(args.place_id)
    name = place.name if place else args.place_id
    moved = state.model_copy(update={"location": args.place_id})
    return ToolResult(moved, f"The user is now at {name} ({args.place_id}).")


def _existing(state: SessionState, task_id: str) -> Task:
    task = state.task(task_id)
    if task is None:  # pragma: no cover - ids were checked by validation
        msg = f"no task with id '{task_id}'"
        raise LookupError(msg)
    return task


def _work_minutes(state: SessionState, plan: DayPlan) -> int:
    return sum(
        (b.end - b.start) // timedelta(minutes=1)
        for b in plan.blocks
        if _existing(state, b.task_id).type is not TaskType.FIXED_EVENT
    )


def _unscheduled_lines(
    state: SessionState, items: tuple[Unscheduled, ...], heading: str
) -> list[str]:
    if not items:
        return []
    return [
        heading,
        *(f"{u.task_id} {_existing(state, u.task_id).title}: {u.reason}" for u in items),
    ]


def _highest_number(state: SessionState) -> int:
    """Highest `T<n>` number handed out so far, counting tasks that have since been deleted."""
    numbers = (int(m[1]) for t in state.tasks if (m := _GENERATED_ID.fullmatch(t.id)))
    return max(state.last_task_number, *numbers, 0)


def _next_number(state: SessionState) -> int:
    return _highest_number(state) + 1
