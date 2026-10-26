"""Whether a task can be done if the user sets off at a given time.

Pure arithmetic over the state and reference data. It answers yes or no with the reasons;
suggesting alternatives is left to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from planner.domain.models import ANYWHERE, Mode, TaskStatus, TaskType

if TYPE_CHECKING:
    from datetime import datetime

    from planner.domain.models import BaseData, SessionState, Task


@dataclass(frozen=True, slots=True)
class Feasibility:
    """The outcome of a feasibility check.

    Attributes:
        feasible: True if nothing stops the task.
        reasons: Every blocker found, phrased for the user; empty when feasible.
        arrive: When the user reaches the task's place, if the route is known.
        finish: When the task would end, if it could be timed.
    """

    feasible: bool
    reasons: tuple[str, ...]
    arrive: datetime | None = None
    finish: datetime | None = None


def check_feasibility(  # noqa: PLR0913 - one argument per input the check depends on
    state: SessionState,
    base: BaseData,
    task: Task,
    *,
    start: datetime,
    from_place: str,
    mode: Mode = Mode.DRIVE,
) -> Feasibility:
    """Check whether `task` can be done if the user leaves `from_place` at `start`.

    Args:
        state: The current session state.
        base: The read-only reference data.
        task: The task to check.
        start: When the user sets off.
        from_place: Place id the user sets off from.
        mode: Travel mode.

    Returns:
        `Feasibility` with every blocker found. A missing duration or route is reported,
        never guessed.
    """
    if task.duration_min is None:
        return Feasibility(feasible=False, reasons=("the task has no duration yet; ask the user",))

    reasons: list[str] = []
    unmet = [
        p
        for p in task.prerequisites
        if (t := state.task(p)) is not None and t.status is TaskStatus.TODO
    ]
    if unmet:
        reasons.append(f"prerequisites not done: {', '.join(unmet)}")

    travel = 0
    if task.place != ANYWHERE:
        leg = base.leg(from_place, task.place, mode)
        if leg is None:
            reasons.append(f"no {mode} route from {from_place} to {task.place} in the travel table")
            return Feasibility(feasible=False, reasons=tuple(reasons))
        travel = leg.minutes + leg.buffer_min
    arrive = start + timedelta(minutes=travel)

    begin = arrive
    if task.earliest_start is not None:
        begin = max(arrive, task.earliest_start)
        if task.type is TaskType.FIXED_EVENT and arrive > task.earliest_start:
            reasons.append(
                f"arrives {arrive:%H:%M}, after the event starts at {task.earliest_start:%H:%M}"
            )
    finish = begin + timedelta(minutes=task.duration_min)

    place = base.place(task.place)
    if place is not None and not place.is_open_during(begin, finish):
        reasons.append(f"{place.name} is not open {begin:%a %H:%M}-{finish:%H:%M}")
    if task.deadline is not None and finish > task.deadline:
        reasons.append(
            f"would finish {finish:%a %H:%M}, after the deadline {task.deadline:%a %H:%M}"
        )
    return Feasibility(not reasons, tuple(reasons), arrive, finish)
