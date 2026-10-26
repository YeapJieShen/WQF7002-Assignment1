from planner.domain.models import ANYWHERE, BaseData, Mode, SessionState, Weekday


def test_base_loads(base: BaseData) -> None:
    assert len(base.places) == 8
    assert base.config.horizon_days == 7
    assert base.config.slot_min == 15


def test_travel_table_only_references_known_places(base: BaseData) -> None:
    for leg in base.travel:
        assert {leg.origin, leg.destination} <= base.place_ids


def test_opening_hours_load(base: BaseData) -> None:
    workshop = next(p for p in base.places if p.id == "workshop")
    weekday, saturday = workshop.hours
    assert weekday.days == {Weekday.MON, Weekday.TUE, Weekday.WED, Weekday.THU, Weekday.FRI}
    assert weekday.closes.strftime("%H:%M") == "18:00"
    assert saturday.days == {Weekday.SAT}


def test_persona_wednesday_is_wed_1540_at_office(state: SessionState) -> None:
    assert state.now.strftime("%a %H:%M") == "Wed 15:40"
    assert state.location == "office"
    assert {t.id for t in state.tasks} == {"T1", "T2", "T3", "T4", "T5", "E1"}


def test_persona_tasks_use_known_places(state: SessionState, base: BaseData) -> None:
    for task in state.tasks:
        assert task.place == ANYWHERE or task.place in base.place_ids


def test_persona_prerequisites_exist(state: SessionState) -> None:
    for task in state.tasks:
        assert set(task.prerequisites) <= state.task_ids


def test_leg_lookup_is_symmetric(base: BaseData) -> None:
    forward = base.leg("home", "um", Mode.DRIVE)
    assert forward is not None
    assert forward.minutes == 25
    assert base.leg("um", "home", Mode.DRIVE) == forward


def test_leg_missing_route_is_none(base: BaseData) -> None:
    assert base.leg("library", "home", Mode.TRANSIT) is None


def test_leg_same_place_is_zero(base: BaseData) -> None:
    leg = base.leg("um", "um", Mode.DRIVE)
    assert leg is not None
    assert leg.minutes == 0
