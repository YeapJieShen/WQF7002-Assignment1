"""Core entities of the planner.

Every entity is immutable: a change produces a new object instead of editing the old one.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from enum import StrEnum
from functools import cached_property
from typing import Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, NaiveDatetime, model_validator

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
    duration_min: int | None = None
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
    minutes: int
    buffer_min: int


class Config(Frozen):
    """Planning parameters.

    Attributes:
        slot_min: Length of one scheduling slot in minutes.
        horizon_days: How many days ahead the agent plans.
        daily_work_cap_min: Maximum planned work per day, in minutes.
        min_break_min: Minimum break between work blocks, in minutes.
    """

    slot_min: int
    horizon_days: int
    daily_work_cap_min: int
    min_break_min: int

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


class Pace(Frozen):
    """How busy the user wants their days. Changed only through the UI sliders.

    Attributes:
        density: 0 for relaxed days up to 1 for packed days.
        distribution: 0 for work front-loaded up to 1 for work spread evenly.
    """

    density: Annotated[float, Field(ge=0, le=1)] = 0.5
    distribution: Annotated[float, Field(ge=0, le=1)] = 0.5


class SessionState(Frozen):
    """Everything about the session that changes: the agent's internal model of the user.

    Attributes:
        now: The current time (simulated clock).
        location: Place id where the user is.
        pace: The user's pace settings.
        tasks: All tasks.
    """

    now: NaiveDatetime
    location: str
    pace: Pace = Pace()
    tasks: tuple[Task, ...] = ()

    @cached_property
    def task_ids(self) -> frozenset[str]:
        """Ids of all tasks."""
        return frozenset(t.id for t in self.tasks)

    def task(self, task_id: str) -> Task | None:
        """Return the task with this id, or None."""
        return next((t for t in self.tasks if t.id == task_id), None)
