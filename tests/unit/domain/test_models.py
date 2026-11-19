import copy
import pickle
from datetime import date
from itertools import combinations

import pytest
from pydantic import ValidationError

from planner.domain.models import (
    BaseData,
    Config,
    Mode,
    OpeningWindow,
    Pace,
    Place,
    SessionState,
    Task,
    TravelLeg,
    Weekday,
)


def test_base_data_is_immutable(base: BaseData) -> None:
    with pytest.raises(ValidationError):
        base.config.slot_min = 9999  # type: ignore[misc]  # assigning to a frozen model must raise
    with pytest.raises(ValidationError):
        base.places[0].name = "x"  # type: ignore[misc]  # assigning to a frozen model must raise


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


def test_every_place_pair_is_reachable_by_car(base: BaseData) -> None:
    ids = [p.id for p in base.places]
    missing = [(a, b) for a, b in combinations(ids, 2) if base.leg(a, b, Mode.DRIVE) is None]
    assert missing == []


def test_task_ids_continue_after_a_deleted_task_in_a_saved_session(state: SessionState) -> None:
    saved = state.model_dump(mode="json")
    saved.pop("last_task_number")
    assert SessionState.model_validate(saved).last_task_number == 0
    assert SessionState.model_validate_json(state.model_dump_json()) == state


@pytest.mark.parametrize("field", ["duration_min"])
@pytest.mark.parametrize("value", [0, -5])
def test_task_duration_must_be_positive(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        Task.model_validate({"id": "T1", "title": "x", "type": "flexible", field: value})


def test_travel_leg_cannot_be_negative() -> None:
    with pytest.raises(ValidationError):
        TravelLeg(origin="a", destination="b", mode=Mode.DRIVE, minutes=-1, buffer_min=0)
    with pytest.raises(ValidationError):
        TravelLeg(origin="a", destination="b", mode=Mode.DRIVE, minutes=1, buffer_min=-1)


def test_config_rejects_zero_slot(base: BaseData) -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({**base.config.model_dump(), "slot_min": 0})


def test_duplicate_place_ids_are_rejected(base: BaseData) -> None:
    data = base.model_dump()
    data["places"] = [*data["places"], data["places"][0]]
    with pytest.raises(ValidationError, match="share an id"):
        BaseData.model_validate(data)


def test_a_route_listed_twice_is_rejected(base: BaseData) -> None:
    data = base.model_dump()
    first = data["travel"][0]
    data["travel"] = [
        *data["travel"],
        {**first, "origin": first["destination"], "destination": first["origin"]},
    ]
    with pytest.raises(ValidationError, match="twice"):
        BaseData.model_validate(data)


def test_duplicate_task_ids_are_rejected(state: SessionState) -> None:
    with pytest.raises(ValidationError, match="share an id"):
        state.with_tasks((*state.tasks, state.tasks[0]))


def test_weekday_limits_cannot_be_mutated_or_leak() -> None:
    pace = Pace(weekday_limit_min={Weekday.MON: 60})
    with pytest.raises(TypeError):
        pace.weekday_limit_min[Weekday.TUE] = 1  # type: ignore[index]  # the read-only mapping must refuse item assignment
    with pytest.raises(TypeError):
        pace.weekday_limit_min.update({Weekday.TUE: 1})  # type: ignore[attr-defined]  # update is blocked; the test checks it raises
    assert Pace().weekday_limit_min == {}
    assert hash(pace) == hash(Pace(weekday_limit_min={Weekday.MON: 60}))


def test_pace_survives_copy_pickle_and_json() -> None:
    pace = Pace(weekday_limit_min={Weekday.MON: 60})
    assert copy.deepcopy(pace) == pace
    assert pickle.loads(pickle.dumps(pace)) == pace  # noqa: S301 - our own object
    assert Pace.model_validate_json(pace.model_dump_json()) == pace
