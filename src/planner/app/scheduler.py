# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
"""Day planning and the "what now" recommendation.

A simplified least-laxity-first scheduler. Laxity (slack) is how much spare time a task has:

    slack = (deadline - now) - (duration + travel)

It is computed when a task is placed, not continuously, and a task is never preempted. Tasks
without a deadline have unlimited slack and come after those with one, ordered by duration
according to `Pace.hardest_first`.

Only flexible tasks are split into chunks; errands and fixed events are atomic. Fixed events
are not scheduled but are treated as walls the other tasks are placed around.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING

from planner.app.feasibility import check_feasibility
from planner.domain.models import ANYWHERE, Mode, TaskStatus, TaskType

if TYPE_CHECKING:
    from datetime import date

    from planner.domain.models import BaseData, SessionState, Task


@dataclass(frozen=True, slots=True)
class Block:
    """A stretch of the day given to one task.

    Attributes:
        task_id: The task being worked on.
        start: When work on it begins.
        end: When work on it ends.
    """

    task_id: str
    start: datetime
    end: datetime


@dataclass(frozen=True, slots=True)
class Unscheduled:
    """A task the planner could not place, or could not recommend.

    Attributes:
        task_id: The task left out.
        reason: Why, phrased for the user.
    """

    task_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class DayPlan:
    """The result of planning one day.

    Attributes:
        blocks: Work and fixed events in time order.
        unscheduled: Tasks that did not get all the time they were due today.
    """

    blocks: tuple[Block, ...]
    unscheduled: tuple[Unscheduled, ...]


@dataclass(frozen=True, slots=True)
class Recommendation:
    """What to do right now.

    Attributes:
        task_id: The recommended task, or None if nothing can be done now.
        arrive: When the user would reach its place.
        finish: When it would be done.
        skipped: Tasks considered and ruled out, each with the reasons.
    """

    task_id: str | None
    arrive: datetime | None
    finish: datetime | None
    skipped: tuple[Unscheduled, ...]


def plan_day(state: SessionState, base: BaseData, day: date) -> DayPlan:
    """Plan one day.

    Args:
        state: The current session state.
        base: The read-only reference data.
        day: The day to plan, inside the planning horizon.

    Returns:
        The blocks in time order, and the tasks that did not fit with the reason for each.
    """
    return _DayPlanner(state, base, day).plan()


def recommend_now(state: SessionState, base: BaseData) -> Recommendation:
    """Recommend the most urgent task that can be done starting now from the current location.

    Args:
        state: The current session state.
        base: The read-only reference data.

    Returns:
        The best feasible task, or no task and the reasons every candidate was ruled out.
        Fixed events are never recommended.
    """
    feasible = []
    skipped = []
    for task in state.tasks:
        if task.status is not TaskStatus.TODO or task.type is TaskType.FIXED_EVENT:
            continue
        result = check_feasibility(state, base, task, start=state.now, from_place=state.location)
        if result.feasible:
            feasible.append((task, result))
        else:
            skipped.append(Unscheduled(task.id, "; ".join(result.reasons)))
    feasible.sort(key=lambda pair: _priority(pair[0], state, base, state.now, state.location))
    if not feasible:
        return Recommendation(None, None, None, tuple(skipped))
    task, result = feasible[0]
    return Recommendation(task.id, result.arrive, result.finish, tuple(skipped))


def _travel_min(base: BaseData, origin: str, destination: str) -> int | None:
    """Minutes by road including buffer; 0 for a task that can be done anywhere."""
    if destination == ANYWHERE:
        return 0
    leg = base.leg(origin, destination, Mode.DRIVE)
    return None if leg is None else leg.minutes + leg.buffer_min


def _priority(
    task: Task, state: SessionState, base: BaseData, now: datetime, origin: str
) -> tuple[bool, timedelta, int]:
    """Sort key: tasks with a deadline first by least slack, then by duration."""
    duration = task.duration_min or 0
    by_duration = -duration if state.pace.hardest_first else duration
    if task.deadline is None:
        return (True, timedelta(0), by_duration)
    travel = _travel_min(base, origin, task.place) or 0
    slack = task.deadline - now - timedelta(minutes=duration + travel)
    return (False, slack, by_duration)


def _ceil_slot(moment: datetime, slot_min: int) -> datetime:
    midnight = datetime.combine(moment.date(), time.min)
    slot = timedelta(minutes=slot_min)
    return midnight + -((midnight - moment) // slot) * slot


@dataclass(frozen=True, slots=True)
class _Wall:
    """A fixed event the other tasks are planned around."""

    task_id: str
    start: datetime
    end: datetime
    place: str


class _DayPlanner:
    """Places tasks one after another through a single day, then reports what was left."""

    def __init__(self, state: SessionState, base: BaseData, day: date) -> None:
        cfg = base.config
        self.state = state
        self.base = base
        self.cap_left = state.pace.limit_on(day)
        self.day_end = datetime.combine(day, cfg.day_end)
        day_start = datetime.combine(day, cfg.day_start)
        self.cursor = _ceil_slot(max(state.now, day_start), cfg.slot_min)
        self.first = self.cursor
        self.location = state.location
        self.walls = deque(sorted(_walls(state, day), key=lambda w: w.start))
        self.workable = [
            t
            for t in state.tasks
            if t.status is TaskStatus.TODO
            and t.type is not TaskType.FIXED_EVENT
            and t.duration_min is not None
        ]
        self.quota = {t.id: self._quota(t, day) for t in self.workable}
        self.placed = {t.id: 0 for t in self.workable}
        self.complete: set[str] = set()
        self.run = 0  # minutes worked since the last long break
        self.last_end: datetime | None = None
        self.blocks: list[Block] = []

    def plan(self) -> DayPlan:
        """Fill the day, then list what did not get its time."""
        while self.cursor < self.day_end:
            self._enter_walls()
            if not self._place_next():
                self._idle()
        self.blocks.extend(
            Block(w.task_id, w.start, w.end) for w in self.walls if w.end > self.state.now
        )
        return DayPlan(tuple(self.blocks), self._unscheduled())

    def _quota(self, task: Task, day: date) -> int:
        """Minutes of this task due on `day`: all of it, or its share if the pace spreads work.

        The share comes from the fraction that should be finished by the end of each day from
        today to the deadline, so a task's daily quotas add up to its duration.
        """
        total = task.duration_min or 0
        if task.type is not TaskType.FLEXIBLE:
            return total
        today = self.state.now.date()
        cfg = self.base.config
        if task.deadline is None:
            last = cfg.horizon_end(self.state.now).date()
        elif task.deadline.time() <= cfg.day_start:
            last = task.deadline.date() - timedelta(days=1)  # nothing can be placed that day
        else:
            last = task.deadline.date()
        days = max((last - today).days, 0) + 1
        offset = (day - today).days
        if offset >= days:
            return total
        return self._done_by(total, offset, days) - self._done_by(total, offset - 1, days)

    def _done_by(self, total: int, offset: int, days: int) -> int:
        """Minutes that should be finished by the end of the day `offset` days from today."""
        if offset < 0:
            return 0
        spread = self.state.pace.distribution
        fraction = (1 - spread) + spread * (offset + 1) / days
        slot = self.base.config.slot_min
        return min(total, -(-round(total * fraction) // slot) * slot)

    def _enter_walls(self) -> None:
        while self.walls and self.walls[0].start <= self.cursor:
            wall = self.walls.popleft()
            if wall.end > self.cursor:
                self.blocks.append(Block(wall.task_id, wall.start, wall.end))
                self.cursor = wall.end
                self.location = wall.place

    def _idle(self) -> None:
        step = self.cursor + timedelta(minutes=self.base.config.slot_min)
        self.cursor = min(step, self.walls[0].start) if self.walls else step

    def _place_next(self) -> bool:
        ready = [t for t in self.workable if self._is_ready(t)]
        ready.sort(key=lambda t: _priority(t, self.state, self.base, self.cursor, self.location))
        for task in ready:
            fit = self._fit(task)
            if fit is not None:
                self._commit(task, *fit)
                return True
        return False

    def _is_ready(self, task: Task) -> bool:
        return self.placed[task.id] < self.quota[task.id] and all(
            self._satisfied(p) for p in task.prerequisites
        )

    def _satisfied(self, task_id: str) -> bool:
        prerequisite = self.state.task(task_id)
        return (
            prerequisite is None
            or prerequisite.status is TaskStatus.DONE
            or task_id in self.complete
        )

    def _fit(self, task: Task) -> tuple[datetime, int] | None:
        """Where the next chunk of `task` would go, or None if it does not fit now."""
        travel = _travel_min(self.base, self.location, task.place)
        if travel is None:
            return None
        begin = self.cursor + timedelta(minutes=travel)
        if task.earliest_start is not None:
            begin = max(begin, _ceil_slot(task.earliest_start, self.base.config.slot_min))
        need = self.quota[task.id] - self.placed[task.id]
        limit = self.state.pace.max_consecutive_min
        if (
            task.type is not TaskType.FLEXIBLE
            and limit is not None
            and self.last_end is not None
            and self._run_at(begin) + need > limit
        ):
            begin = max(begin, self.last_end + timedelta(minutes=self.state.pace.long_break_min))
        room = self._room_end(task) - begin
        if task.type is TaskType.FLEXIBLE:
            slot = self.base.config.slot_min
            room_slots = max(room // timedelta(minutes=slot), 0) * slot
            headroom = need if limit is None else limit - self._run_at(begin)
            length = min(need, self.base.config.max_chunk_min, self.cap_left, room_slots, headroom)
            if length < min(slot, need):
                return None
        else:
            length = need
            if length > self.cap_left or timedelta(minutes=length) > room:
                return None
        place = self.base.place(task.place)
        if place is not None and not place.is_open_during(begin, begin + timedelta(minutes=length)):
            return None
        return begin, length

    def _run_at(self, begin: datetime) -> int:
        """Minutes worked in a row up to `begin`: reset by a gap as long as a long break."""
        long_break = timedelta(minutes=self.state.pace.long_break_min)
        if self.last_end is None or begin - self.last_end >= long_break:
            return 0
        return self.run

    def _room_end(self, task: Task) -> datetime:
        """Latest moment work on `task` may end: day end, deadline, or leaving for the next wall."""
        end = self.day_end
        if task.deadline is not None:
            end = min(end, task.deadline)
        if self.walls:
            wall = self.walls[0]
            origin = self.location if task.place == ANYWHERE else task.place
            back = _travel_min(self.base, origin, wall.place) or 0
            end = min(end, wall.start - timedelta(minutes=back))
        return end

    def _commit(self, task: Task, begin: datetime, length: int) -> None:
        end = begin + timedelta(minutes=length)
        self.blocks.append(Block(task.id, begin, end))
        self.placed[task.id] += length
        self.cap_left -= length
        self.run = self._run_at(begin) + length
        self.last_end = end
        limit = self.state.pace.max_consecutive_min
        long_break = limit is not None and self.run >= limit
        pause = self.state.pace.long_break_min if long_break else self.base.config.min_break_min
        self.cursor = end + timedelta(minutes=pause)
        if task.place != ANYWHERE:
            self.location = task.place
        if self.placed[task.id] >= (task.duration_min or 0):
            self.complete.add(task.id)

    def _unscheduled(self) -> tuple[Unscheduled, ...]:
        left = [
            Unscheduled(t.id, self._why(t))
            for t in self.workable
            if self.placed[t.id] < self.quota[t.id]
        ]
        missing = [
            Unscheduled(t.id, "has no duration yet; ask the user")
            for t in self.state.tasks
            if t.status is TaskStatus.TODO
            and t.type is not TaskType.FIXED_EVENT
            and t.duration_min is None
        ]
        unstarted = [
            Unscheduled(t.id, "has no start time yet; ask the user")
            for t in self.state.tasks
            if t.status is TaskStatus.TODO
            and t.type is TaskType.FIXED_EVENT
            and t.earliest_start is None
        ]
        return tuple(left + missing + unstarted)

    def _why(self, task: Task) -> str:
        if unmet := [p for p in task.prerequisites if not self._satisfied(p)]:
            return f"waiting on {', '.join(unmet)}"
        if task.earliest_start is not None and task.earliest_start >= self.day_end:
            return f"not available until {task.earliest_start:%a %H:%M}"
        if _travel_min(self.base, self.state.location, task.place) is None:
            return f"no route from {self.state.location} to {task.place} in the travel table"
        if task.deadline is not None and task.deadline <= self.first:
            return "its deadline has already passed"
        placed = self.placed[task.id]
        done = f" ({placed} of {self.quota[task.id]} min placed)" if placed else ""
        if task.deadline is not None and task.deadline <= self.day_end:
            return f"does not fit before its deadline{done}"
        return f"no room left today{done}"


def _walls(state: SessionState, day: date) -> list[_Wall]:
    walls = []
    for task in state.tasks:
        start = task.earliest_start
        if (
            task.type is TaskType.FIXED_EVENT
            and task.status is TaskStatus.TODO
            and start is not None
            and start.date() == day
        ):
            end = start + timedelta(minutes=task.duration_min or 0)
            walls.append(_Wall(task.id, start, end, task.place))
    return walls
