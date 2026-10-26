"""What each tool does once its call has been validated.

Handlers trust their arguments: ids, places and dates were checked by `planner.app.validation`.
They never edit the state; a change returns a new one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from planner.app.validation import DELETE_ALL
from planner.domain.models import ANYWHERE, SessionState, Task, TaskType

if TYPE_CHECKING:
    from planner.app.tools.args import AddTaskArgs, DeleteTaskArgs, TravelTimeArgs, UpdateTaskArgs
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
    task = Task(
        id=_next_id(state),
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
        state.with_tasks((*state.tasks, task)), f"Added {task.id}: {task.title}.{note}"
    )


def update_task(state: SessionState, args: UpdateTaskArgs) -> ToolResult:
    """Change the given fields of a task."""
    changes = args.fields.model_dump(exclude_none=True)
    tasks = tuple(t.model_copy(update=changes) if t.id == args.task_id else t for t in state.tasks)
    return ToolResult(state.with_tasks(tasks), f"Updated {args.task_id}: {', '.join(changes)}.")


def delete_task(state: SessionState, args: DeleteTaskArgs) -> ToolResult:
    """Delete one task, or all of them. Confirmation is the caller's job, not this function's."""
    if args.task_id == DELETE_ALL:
        return ToolResult(state.with_tasks(()), f"Deleted all {len(state.tasks)} tasks.")
    remaining = tuple(
        t.model_copy(
            update={"prerequisites": tuple(p for p in t.prerequisites if p != args.task_id)}
        )
        for t in state.tasks
        if t.id != args.task_id
    )
    return ToolResult(state.with_tasks(remaining), f"Deleted {args.task_id}.")


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


def _next_id(state: SessionState) -> str:
    numbers = (int(m[1]) for t in state.tasks if (m := _GENERATED_ID.fullmatch(t.id)))
    return f"T{max(numbers, default=0) + 1}"
