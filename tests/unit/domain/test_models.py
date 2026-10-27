from datetime import date

import pytest
from pydantic import ValidationError

from planner.domain.models import BaseData, OpeningWindow, Pace, Place, SessionState, Weekday


def test_base_data_is_immutable(base: BaseData) -> None:
    with pytest.raises(ValidationError):
        base.config.slot_min = 9999  # type: ignore[misc]
    with pytest.raises(ValidationError):
        base.places[0].name = "x"  # type: ignore[misc]


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Pace.model_validate({"distribution": 0.5, "speed": 1})


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_pace_is_bounded(value: float) -> None:
    with pytest.raises(ValidationError):
        Pace(distribution=value)


def test_pace_limit_uses_weekday_override() -> None:
    pace = Pace(daily_limit_min=480, weekday_limit_min={Weekday.SAT: 120})
    assert pace.limit_on(date(2026, 10, 31)) == 120  # Saturday
    assert pace.limit_on(date(2026, 10, 30)) == 480  # Friday


def test_unknown_weekday_rejected() -> None:
    with pytest.raises(ValidationError):
        OpeningWindow.model_validate({"days": ["funday"], "opens": "09:00", "closes": "17:00"})


def test_place_without_hours_is_always_open() -> None:
    assert Place(id="home", name="Home", category="home").hours == ()


def test_state_task_lookup(state: SessionState) -> None:
    task = state.task("T2")
    assert task is not None
    assert task.title == "Car service"
    assert state.task("nope") is None


def test_horizon_end_is_seven_days_after_now(state: SessionState, base: BaseData) -> None:
    assert (base.config.horizon_end(state.now) - state.now).days == 7


def test_travel_table_with_unknown_place_rejected() -> None:
    config = {
        "slot_min": 15,
        "horizon_days": 7,
        "min_break_min": 15,
        "max_chunk_min": 120,
        "day_start": "08:00",
        "day_end": "22:00",
    }
    data = {
        "config": config,
        "places": [{"id": "home", "name": "Home", "category": "home"}],
        "travel": [
            {
                "origin": "home",
                "destination": "mars",
                "mode": "drive",
                "minutes": 9,
                "buffer_min": 1,
            }
        ],
    }
    with pytest.raises(ValidationError, match="unknown place"):
        BaseData.model_validate(data)
