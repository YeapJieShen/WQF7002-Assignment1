from datetime import datetime

from planner.app.tools.args import (
    AddTaskArgs,
    DeleteTaskArgs,
    TaskPatch,
    TravelTimeArgs,
    UpdateTaskArgs,
)
from planner.app.tools.handlers import add_task, delete_task, travel_time, update_task
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
