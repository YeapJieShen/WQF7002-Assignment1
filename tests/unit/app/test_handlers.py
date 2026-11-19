# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
from datetime import date, datetime

import pytest
from pydantic import ValidationError

from planner.app.tools.args import (
    AddTaskArgs,
    CheckFeasibilityArgs,
    DeleteTaskArgs,
    PlanDayArgs,
    SetLocationArgs,
    TaskPatch,
    TravelTimeArgs,
    UpdateTaskArgs,
)
from planner.app.tools.handlers import (
    add_task,
    check_feasibility,
    delete_task,
    plan_day,
    recommend_now,
    set_location,
    travel_time,
    update_task,
)
from planner.domain.models import BaseData, Mode, SessionState, TaskStatus, TaskType


def test_add_task_assigns_next_id_and_defaults(state: SessionState) -> None:
    result = add_task(state, AddTaskArgs(title="Post parcel", duration_min=20, place="post_office"))
    added = result.state.task("T6")
    assert added is not None
    assert (added.type, added.place, added.status) == (TaskType.FLEXIBLE, "post_office", "todo")
    assert len(result.state.tasks) == len(state.tasks) + 1
    assert "T6" in result.message
    assert "ask the user" not in result.message


def test_add_task_without_duration_asks_for_one(state: SessionState) -> None:
    result = add_task(state, AddTaskArgs(title="Dentist"))
    assert "no duration" in result.message
    added = result.state.task("T6")
    assert added is not None
    assert (added.duration_min, added.place) == (None, "anywhere")


def test_add_task_does_not_reuse_ids_of_deleted_tasks(state: SessionState) -> None:
    without_t3 = delete_task(state, DeleteTaskArgs(task_id="T3")).state
    assert add_task(without_t3, AddTaskArgs(title="x")).state.task("T6") is not None


def test_add_task_on_empty_state_starts_at_t1(state: SessionState) -> None:
    result = add_task(state.with_tasks(()), AddTaskArgs(title="x"))
    assert result.state.task("T1") is not None


def test_update_changes_only_given_fields(state: SessionState) -> None:
    patch = TaskPatch(duration_min=30, deadline=datetime(2026, 10, 29, 12, 0))
    result = update_task(state, UpdateTaskArgs(task_id="T3", fields=patch))
    before, after = state.task("T3"), result.state.task("T3")
    assert before is not None
    assert after is not None
    assert after.duration_min == 30
    assert after.deadline == datetime(2026, 10, 29, 12, 0)
    assert (after.title, after.place) == (before.title, before.place)
    assert result.state.task("T1") == state.task("T1")
    assert "duration_min" in result.message


def test_update_can_mark_done(state: SessionState) -> None:
    patch = TaskPatch(status=TaskStatus.DONE)
    after = update_task(state, UpdateTaskArgs(task_id="T4", fields=patch)).state.task("T4")
    assert after is not None
    assert after.status is TaskStatus.DONE


def test_delete_one_removes_it_and_dangling_prerequisites(state: SessionState) -> None:
    result = delete_task(state, DeleteTaskArgs(task_id="T2"))
    assert result.state.task("T2") is None
    assert "T2" not in result.state.task_ids
    t5 = result.state.task("T5")
    assert t5 is not None
    assert t5.prerequisites == ()


def test_delete_all(state: SessionState) -> None:
    result = delete_task(state, DeleteTaskArgs(task_id="all"))
    assert result.state.tasks == ()
    assert "6" in result.message


def test_travel_time_found(state: SessionState, base: BaseData) -> None:
    args = TravelTimeArgs(from_place="office", to_place="um", mode=Mode.TRANSIT)
    result = travel_time(state, base, args)
    assert "40 min" in result.message
    assert "5 min buffer" in result.message
    assert result.state is state


def test_travel_time_missing_route_is_not_estimated(state: SessionState, base: BaseData) -> None:
    args = TravelTimeArgs(from_place="grocery", to_place="um", mode=Mode.TRANSIT)
    assert "No transit route" in travel_time(state, base, args).message


def test_check_feasibility_says_no_with_reasons(state: SessionState, base: BaseData) -> None:
    args = CheckFeasibilityArgs(
        task_id="T2", start=datetime(2026, 10, 28, 15, 40), from_place="office"
    )
    message = check_feasibility(state, base, args).message
    assert message.startswith("T2 does not fit")
    assert "opening hours on Wed: 09:00-18:00" in message


def test_check_feasibility_says_yes_with_times(state: SessionState, base: BaseData) -> None:
    args = CheckFeasibilityArgs(
        task_id="T3", start=datetime(2026, 10, 28, 15, 40), from_place="office"
    )
    expected = "T3 fits: arrive Wed 16:00, finish Wed 16:45."
    assert check_feasibility(state, base, args).message == expected


def test_plan_day_lists_blocks_total_and_leftovers(state: SessionState, base: BaseData) -> None:
    message = plan_day(state, base, PlanDayArgs(date=date(2026, 10, 28))).message
    assert message.startswith("Plan for Wed 2026-10-28:")
    assert "15:45-16:00 T4 Reply to supervisor" in message
    assert "19:00-22:00 E1 Class at UM" in message
    assert "Planned work: 120 min." in message  # class is not counted as work
    assert "Not scheduled:\n" in message
    assert "T2 Car service: does not fit before its deadline" in message


def test_plan_day_with_nothing_to_do(state: SessionState, base: BaseData) -> None:
    message = plan_day(state.with_tasks(()), base, PlanDayArgs(date=date(2026, 10, 29))).message
    assert "Nothing scheduled." in message
    assert "Not scheduled" not in message


def test_recommend_now_names_the_task_and_what_was_ruled_out(
    state: SessionState, base: BaseData
) -> None:
    message = recommend_now(state, base).message
    assert message.startswith("Do T4 Reply to supervisor now: arrive 15:40, finish 15:55.")
    assert "Ruled out:\nT2 Car service:" in message


def test_recommend_now_when_nothing_is_possible(state: SessionState, base: BaseData) -> None:
    only_car = state.with_tasks(tuple(t for t in state.tasks if t.id == "T2"))
    assert recommend_now(only_car, base).message.startswith("Nothing can be done right now.")


def test_set_location_moves_the_user_and_keeps_the_tasks(
    state: SessionState, base: BaseData
) -> None:
    result = set_location(state, base, SetLocationArgs(place_id="library"))
    assert result.state.location == "library"
    assert result.state.tasks == state.tasks
    assert result.message == "The user is now at Faculty library (library)."


def test_set_location_names_an_unlisted_place_by_its_id(
    state: SessionState, base: BaseData
) -> None:
    assert "(mars)" in set_location(state, base, SetLocationArgs(place_id="mars")).message


def test_deleting_the_newest_task_does_not_free_its_id(state: SessionState) -> None:
    added = add_task(state, AddTaskArgs(title="x")).state  # T6
    gone = delete_task(added, DeleteTaskArgs(task_id="T6")).state
    assert add_task(gone, AddTaskArgs(title="y")).state.task("T7") is not None


def test_deleting_all_tasks_does_not_restart_the_numbering(state: SessionState) -> None:
    cleared = delete_task(state, DeleteTaskArgs(task_id="all")).state
    assert add_task(cleared, AddTaskArgs(title="y")).state.task("T6") is not None


def test_update_revalidates_the_merged_task(state: SessionState) -> None:
    unchecked = TaskPatch.model_construct(duration_min=-5)  # skips the patch's own checks
    args = UpdateTaskArgs.model_construct(task_id="T3", fields=unchecked)
    with pytest.raises(ValidationError):
        update_task(state, args)
