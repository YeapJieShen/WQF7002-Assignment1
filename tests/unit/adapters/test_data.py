# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
import pytest

from planner.adapters.data import (
    list_personas,
    load_persona,
    load_rules,
    persona_choices,
    persona_label,
)
from planner.domain.models import ANYWHERE, BaseData, Mode, SessionState, Weekday


def test_rules_load_and_are_not_empty() -> None:
    assert "planning assistant" in load_rules()


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


def test_personas_are_listed_by_file_name() -> None:
    assert "persona_wednesday" in list_personas()


@pytest.mark.parametrize("name", list_personas())
def test_every_persona_loads_with_known_places_and_tasks(name: str, base: BaseData) -> None:
    persona = load_persona(name)
    ids = {t.id for t in persona.tasks}
    assert persona.location in base.place_ids
    assert all(t.place in base.place_ids | {ANYWHERE} for t in persona.tasks)
    assert all(set(t.prerequisites) <= ids for t in persona.tasks)


def test_there_is_a_second_persona() -> None:
    assert "persona_saturday" in list_personas()


def test_a_persona_is_labelled_by_its_start_and_every_persona_is_offered() -> None:
    assert persona_label(load_persona("persona_wednesday")) == "Wednesday 28 Oct, 15:40"
    choices = persona_choices()
    assert ("Persona 2: Saturday 31 Oct, 09:30", "persona_saturday") in choices
    assert {name for _, name in choices} == set(list_personas())
    assert choices[0][0] == "Persona 1: Wednesday 28 Oct, 15:40"  # earliest start first
