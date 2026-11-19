"""Core entities of the planner.

Every entity is immutable: a change produces a new object instead of editing the old one.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from functools import cached_property
from typing import Annotated, Any, Final, NoReturn, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    NaiveDatetime,
    PlainSerializer,
    model_validator,
)

ANYWHERE: Final = "anywhere"
"""Place value for a task that can be done at any location."""


class Frozen(BaseModel):
    """Base class for all entities: immutable, and unknown fields are an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class TaskType(StrEnum):
    """What kind of commitment a task is."""

    FIXED_EVENT = "fixed_event"
    FLEXIBLE = "flexible"
    ERRAND = "errand"


class TaskStatus(StrEnum):
    """Whether a task still needs doing."""

    TODO = "todo"
    DONE = "done"


class Mode(StrEnum):
    """How the user travels between places."""

    DRIVE = "drive"
    TRANSIT = "transit"


class Weekday(StrEnum):
    """Day of the week, as written in the opening-hours data."""

    MON = "mon"
    TUE = "tue"
    WED = "wed"
    THU = "thu"
    FRI = "fri"
    SAT = "sat"
    SUN = "sun"


class Task(Frozen):
    """Something the user has to do.

    For a fixed event, `earliest_start` is its start time.

    Attributes:
        id: Unique task identifier, assigned by the system.
        title: Short description.
        type: Kind of commitment.
        duration_min: Estimated effort in minutes; None until the user gives one.
        deadline: Latest acceptable finish time, if any.
        place: A place id, or `ANYWHERE`.
        earliest_start: Earliest acceptable start time, if any.
        prerequisites: Ids of tasks that must be done first.
        status: Whether the task is still to do.
    """

    id: str
    title: str
    type: TaskType
    duration_min: Annotated[int, Field(gt=0)] | None = None
    deadline: NaiveDatetime | None = None
    place: str = ANYWHERE
    earliest_start: NaiveDatetime | None = None
    prerequisites: tuple[str, ...] = ()
    status: TaskStatus = TaskStatus.TODO


class OpeningWindow(Frozen):
    """Hours a place is open on a set of weekdays.

    Attributes:
        days: Weekdays this window applies to.
        opens: Opening time.
        closes: Closing time.
    """

    days: frozenset[Weekday]
    opens: time
    closes: time


class Place(Frozen):
    """A known location.

    Attributes:
        id: Unique place identifier used by tasks and the travel table.
        name: Human-readable name.
        category: Kind of place, such as "shop" or "service".
        hours: Opening windows; empty means always open.
    """

    id: str
    name: str
    category: str
    hours: tuple[OpeningWindow, ...] = ()

    def is_open_during(self, start: datetime, end: datetime) -> bool:
        """Tell whether the place is open for the whole interval.

        Args:
            start: When the visit begins.
            end: When the visit ends.

        Returns:
            True if the place has no listed hours, or one opening window covers the interval.
            A visit that spans midnight never fits.
        """
        if not self.hours:
            return True
        if start.date() != end.date():
            return False
        day = tuple(Weekday)[start.weekday()]
        return any(
            day in w.days and w.opens <= start.time() and end.time() <= w.closes for w in self.hours
        )

    def hours_on(self, day: date) -> str:
        """Describe the opening hours on `day`, such as "09:00-18:00", or "closed"."""
        weekday = tuple(Weekday)[day.weekday()]
        windows = [f"{w.opens:%H:%M}-{w.closes:%H:%M}" for w in self.hours if weekday in w.days]
        return ", ".join(windows) or "closed"


class TravelLeg(Frozen):
    """One row of the travel table.

    Attributes:
        origin: Place id travelled from.
        destination: Place id travelled to.
        mode: Travel mode.
        minutes: Travel time in minutes.
        buffer_min: Extra minutes for parking or walking at the destination.
    """

    origin: str
    destination: str
    mode: Mode
    minutes: Annotated[int, Field(ge=0)]
    buffer_min: Annotated[int, Field(ge=0)]


class Config(Frozen):
    """Planning parameters.

    Attributes:
        slot_min: Length of one scheduling slot in minutes.
        horizon_days: How many days ahead the agent plans.
        min_break_min: Minimum break between work blocks, in minutes.
        max_chunk_min: Longest single block a flexible task is split into, in minutes.
        day_start: When the planning day begins.
        day_end: When the planning day ends.
    """

    slot_min: Annotated[int, Field(gt=0)]
    horizon_days: Annotated[int, Field(gt=0)]
    min_break_min: Annotated[int, Field(ge=0)]
    max_chunk_min: Annotated[int, Field(gt=0)]
    day_start: time
    day_end: time

    def horizon_end(self, now: datetime) -> datetime:
        """Return the last moment the agent plans for, counted from `now`."""
        return now + timedelta(days=self.horizon_days)


class BaseData(Frozen):
    """Read-only reference data: places, travel table and planning parameters.

    Attributes:
        places: All known places.
        travel: The travel table.
        config: Planning parameters.
    """

    places: tuple[Place, ...]
    travel: tuple[TravelLeg, ...]
    config: Config

    @model_validator(mode="after")
    def _ids_and_routes_are_unique(self) -> Self:
        if len(self.place_ids) != len(self.places):
            msg = "two places share an id"
            raise ValueError(msg)
        routes = [(frozenset((leg.origin, leg.destination)), leg.mode) for leg in self.travel]
        if len(set(routes)) != len(routes):
            msg = "the travel table lists the same route twice"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _travel_uses_known_places(self) -> Self:
        for leg in self.travel:
            if not {leg.origin, leg.destination} <= self.place_ids:
                msg = f"travel table names an unknown place: {leg.origin} -> {leg.destination}"
                raise ValueError(msg)
        return self

    def place(self, place_id: str) -> Place | None:
        """Return the place with this id, or None."""
        return next((p for p in self.places if p.id == place_id), None)

    @cached_property
    def place_ids(self) -> frozenset[str]:
        """Ids of all known places."""
        return frozenset(p.id for p in self.places)

    def leg(self, origin: str, destination: str, mode: Mode) -> TravelLeg | None:
        """Look up travel between two places, in either direction.

        Args:
            origin: Place id travelled from.
            destination: Place id travelled to.
            mode: Travel mode.

        Returns:
            The table row, a zero-minute leg if both places are the same, or None if the
            route is not in the table. A missing route is never guessed.
        """
        if origin == destination:
            return TravelLeg(
                origin=origin, destination=destination, mode=mode, minutes=0, buffer_min=0
            )
        endpoints = {origin, destination}
        for leg in self.travel:
            if leg.mode == mode and {leg.origin, leg.destination} == endpoints:
                return leg
        return None


Minutes = Annotated[int, Field(ge=0, le=24 * 60)]


def _blocked(*_args: object, **_kwargs: object) -> NoReturn:
    msg = "FrozenDict is read-only"
    raise TypeError(msg)


class FrozenDict[K, V](dict[K, V]):
    """A read-only, hashable dict, so a frozen model holding one is really immutable."""

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _blocked

    def __hash__(self) -> int:  # type: ignore[override]  # dict sets __hash__ to None; this read-only dict opts back in
        """Hash by content, like a frozenset of the items."""
        return hash(frozenset(self.items()))

    def __reduce__(self) -> tuple[Any, ...]:
        """Rebuild through the constructor, which bypasses the blocked mutators."""
        return (type(self), (dict(self),))


WeekdayLimits = Annotated[
    Mapping[Weekday, Minutes], AfterValidator(FrozenDict), PlainSerializer(dict)
]


class Pace(Frozen):
    """How the user wants their work paced. Changed only through the UI, never by the model.

    Attributes:
        daily_limit_min: Most work to plan on a day, in minutes.
        max_consecutive_min: Longest run of work before a long break; None turns this off.
        long_break_min: Length of that long break, in minutes.
        weekday_limit_min: Per-weekday replacements for `daily_limit_min`.
        distribution: 0 for work front-loaded up to 1 for work spread evenly.
        hardest_first: Among tasks without a deadline, do longer ones first (else shorter first).
    """

    daily_limit_min: Minutes = 480
    max_consecutive_min: Minutes | None = 240
    long_break_min: Minutes = 45
    weekday_limit_min: WeekdayLimits = Field(default_factory=dict, validate_default=True)
    distribution: Annotated[float, Field(ge=0, le=1)] = 0.5
    hardest_first: bool = False

    def limit_on(self, day: date) -> int:
        """Return the most work, in minutes, to plan on this day."""
        return self.weekday_limit_min.get(tuple(Weekday)[day.weekday()], self.daily_limit_min)


class SessionState(Frozen):
    """Everything about the session that changes: the agent's internal model of the user.

    Attributes:
        now: The current time (simulated clock).
        location: Place id where the user is.
        pace: The user's pace settings.
        tasks: All tasks.
        last_task_number: Highest `T<n>` number ever handed out, so a deleted id is never reused.
            It only matters when it exceeds every number still in use; 0 means "none recorded".
    """

    now: NaiveDatetime
    location: str
    pace: Pace = Pace()
    tasks: tuple[Task, ...] = ()
    last_task_number: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def _task_ids_are_unique(self) -> Self:
        if len(self.task_ids) != len(self.tasks):
            msg = "two tasks share an id"
            raise ValueError(msg)
        return self

    @cached_property
    def task_ids(self) -> frozenset[str]:
        """Ids of all tasks."""
        return frozenset(t.id for t in self.tasks)

    def with_tasks(
        self, tasks: tuple[Task, ...], *, last_task_number: int | None = None
    ) -> SessionState:
        """Return a copy with a different task list, and optionally a new id high-water mark.

        Built from the fields rather than `model_copy`, which would carry over the cached
        `task_ids` of the old list.
        """
        fields = {name: getattr(self, name) for name in type(self).model_fields}
        fields["tasks"] = tasks
        if last_task_number is not None:
            fields["last_task_number"] = last_task_number
        return SessionState(**fields)

    def task(self, task_id: str) -> Task | None:
        """Return the task with this id, or None."""
        return next((t for t in self.tasks if t.id == task_id), None)
