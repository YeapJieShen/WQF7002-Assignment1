"""The session snapshot the model reads before each reply.

The fixed rules live in `resources/rules.md` and form the first message, which never changes,
so the backend can reuse its cached prefix. This snapshot changes with every edit or tick of
the clock, so it goes last, after the conversation, to keep that prefix intact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from planner.domain.models import BaseData, SessionState, Task


def session_snapshot(state: SessionState, base: BaseData) -> str:
    """Describe the session as it is now.

    Args:
        state: The current session.
        base: Reference data: places and planning parameters.

    Returns:
        The clock, location, pace, known places and tasks.
    """
    config = base.config
    pace = state.pace
    here = base.place(state.location)
    places = "\n".join(f"- {p.id}: {p.name} ({p.category})" for p in base.places)
    tasks = "\n".join(_describe(t) for t in state.tasks) or "- none"
    return (
        "Current session:\n"
        f"Now: {state.now:%A %Y-%m-%d %H:%M}\n"
        f"The user is at: {here.name if here else state.location} ({state.location})\n"
        f"You plan up to {config.horizon_days} days ahead. The day runs "
        f"{config.day_start:%H:%M} to {config.day_end:%H:%M}.\n"
        f"The user's daily limit is {pace.daily_limit_min} minutes of work.\n\n"
        f"Known places:\n{places}\n\n"
        f"Tasks:\n{tasks}\n"
    )


def _describe(task: Task) -> str:
    parts = [
        task.type.value,
        f"{task.duration_min} min" if task.duration_min is not None else "duration unknown",
        f"at {task.place}",
    ]
    if task.earliest_start:
        parts.append(f"from {task.earliest_start:%a %m-%d %H:%M}")
    if task.deadline:
        parts.append(f"by {task.deadline:%a %m-%d %H:%M}")
    if task.prerequisites:
        parts.append(f"after {', '.join(task.prerequisites)}")
    return f"- {task.id} {task.title} [{task.status.value}]: {', '.join(parts)}"
