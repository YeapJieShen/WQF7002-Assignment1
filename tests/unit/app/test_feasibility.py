from datetime import datetime

import pytest

from planner.app.feasibility import Feasibility, check_feasibility
from planner.domain.models import BaseData, Mode, SessionState, Task, TaskStatus, TaskType

OFFICE_NOW = datetime(2026, 10, 28, 15, 40)


def _check(
    state: SessionState,
    base: BaseData,
    task_id: str,
    start: datetime = OFFICE_NOW,
    from_place: str = "office",
) -> Feasibility:
    task = state.task(task_id)
    assert task is not None
    return check_feasibility(state, base, task, start=start, from_place=from_place)


def test_car_service_today_does_not_fit(state: SessionState, base: BaseData) -> None:
    result = _check(state, base, "T2")
    assert not result.feasible
    assert result.finish == datetime(2026, 10, 28, 18, 10)
    assert "Car workshop opening hours on Wed: 09:00-18:00; the visit would need 16:10-18:10" in (
        result.reasons
    )
    assert any("deadline" in r for r in result.reasons)


def test_car_service_tomorrow_fits_the_hours(state: SessionState, base: BaseData) -> None:
    result = _check(state, base, "T2", start=datetime(2026, 10, 29, 9, 0))
    assert not any("opening hours" in r for r in result.reasons)
    assert result.arrive == datetime(2026, 10, 29, 9, 30)


def test_groceries_fit_now(state: SessionState, base: BaseData) -> None:
    result = _check(state, base, "T3")
    assert result.feasible
    assert result.reasons == ()
    assert result.arrive == datetime(2026, 10, 28, 16, 0)
    assert result.finish == datetime(2026, 10, 28, 16, 45)


def test_anywhere_task_has_no_travel(state: SessionState, base: BaseData) -> None:
    result = _check(state, base, "T4")
    assert result.feasible
    assert result.arrive == OFFICE_NOW


def test_unmet_prerequisite_blocks(state: SessionState, base: BaseData) -> None:
    result = _check(state, base, "T5", start=datetime(2026, 10, 30, 9, 0), from_place="workshop")
    assert any("prerequisites not done: T2" in r for r in result.reasons)


def test_done_prerequisite_does_not_block(state: SessionState, base: BaseData) -> None:
    done = tuple(
        t.model_copy(update={"status": TaskStatus.DONE}) if t.id == "T2" else t for t in state.tasks
    )
    later = state.model_copy(update={"tasks": done})
    result = _check(later, base, "T5", start=datetime(2026, 10, 30, 9, 0), from_place="home")
    assert not any("prerequisites" in r for r in result.reasons)


def test_missing_duration_is_reported(state: SessionState, base: BaseData) -> None:
    task = Task(id="T9", title="Dentist", type=TaskType.ERRAND)
    result = check_feasibility(state, base, task, start=OFFICE_NOW, from_place="office")
    assert not result.feasible
    assert result.arrive is None


def test_missing_route_is_reported(state: SessionState, base: BaseData) -> None:
    task = state.task("T2")
    assert task is not None
    result = check_feasibility(
        state, base, task, start=OFFICE_NOW, from_place="office", mode=Mode.TRANSIT
    )
    assert not result.feasible
    assert "no transit route" in result.reasons[0]


def test_fixed_event_late_arrival(state: SessionState, base: BaseData) -> None:
    result = _check(state, base, "E1", start=datetime(2026, 10, 28, 18, 45))
    assert any("after the event starts" in r for r in result.reasons)


def test_fixed_event_early_arrival_waits(state: SessionState, base: BaseData) -> None:
    result = _check(state, base, "E1", start=datetime(2026, 10, 28, 16, 0))
    assert result.feasible
    assert result.finish == datetime(2026, 10, 28, 22, 0)


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (datetime(2026, 10, 28, 9, 0), datetime(2026, 10, 28, 18, 0), True),
        (datetime(2026, 10, 28, 8, 59), datetime(2026, 10, 28, 12, 0), False),
        (datetime(2026, 10, 28, 17, 0), datetime(2026, 10, 28, 18, 1), False),
        (datetime(2026, 10, 31, 9, 0), datetime(2026, 10, 31, 13, 0), True),
        (datetime(2026, 11, 1, 9, 0), datetime(2026, 11, 1, 10, 0), False),
        (datetime(2026, 10, 28, 17, 0), datetime(2026, 10, 29, 9, 0), False),
    ],
)
def test_workshop_hours(base: BaseData, start: datetime, end: datetime, *, expected: bool) -> None:
    workshop = base.place("workshop")
    assert workshop is not None
    assert workshop.is_open_during(start, end) is expected


def test_place_without_hours_is_always_open(base: BaseData) -> None:
    home = base.place("home")
    assert home is not None
    assert home.is_open_during(datetime(2026, 10, 28, 3, 0), datetime(2026, 10, 29, 3, 0))


def test_unknown_place_lookup(base: BaseData) -> None:
    assert base.place("nowhere") is None


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (datetime(2026, 10, 31), "09:00-13:00"),
        (datetime(2026, 11, 1), "closed"),
    ],
)
def test_hours_on_names_the_windows_or_closed(base: BaseData, day: datetime, expected: str) -> None:
    workshop = base.place("workshop")
    assert workshop is not None
    assert workshop.hours_on(day.date()) == expected
