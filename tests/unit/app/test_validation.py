from datetime import datetime
from typing import Any

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from planner.app.tools.args import (
    AddTaskArgs,
    DeleteTaskArgs,
    PlanDayArgs,
    RecommendNowArgs,
    TravelTimeArgs,
    UpdateTaskArgs,
)
from planner.app.validation import (
    Accepted,
    Rejected,
    creates_cycle,
    reject_malformed,
    validate_call,
)
from planner.domain.errors import ErrorCode
from planner.domain.models import BaseData, SessionState, Task, TaskType
from tests.conftest import Codes

# --- accepted calls --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "args", "expected_type"),
    [
        ("recommend_now", {}, RecommendNowArgs),
        (
            "travel_time",
            {"from_place": "home", "to_place": "um", "mode": "drive"},
            TravelTimeArgs,
        ),
        (
            "check_feasibility",
            {"task_id": "T2", "start": "2026-10-28T17:15", "from_place": "office"},
            None,
        ),
        ("plan_day", {"date": "2026-10-29"}, PlanDayArgs),
        ("delete_task", {"task_id": "T4"}, DeleteTaskArgs),
        ("delete_task", {"task_id": "all"}, DeleteTaskArgs),
        ("update_task", {"task_id": "T4", "fields": {"status": "done"}}, UpdateTaskArgs),
        (
            "add_task",
            {
                "title": "Collect parcel",
                "duration_min": 20,
                "place": "post_office",
                "deadline": "2026-10-30T17:00",
            },
            AddTaskArgs,
        ),
        ("add_task", {"title": "Think about it"}, AddTaskArgs),  # essentials missing: tool asks
    ],
)
def test_accepted(
    name: str,
    args: dict[str, Any],
    expected_type: type | None,
    state: SessionState,
    base: BaseData,
) -> None:
    verdict = validate_call(name, args, state, base)
    assert isinstance(verdict, Accepted), verdict
    if expected_type is not None:
        assert isinstance(verdict.args, expected_type)


def test_accepted_args_are_typed(state: SessionState, base: BaseData) -> None:
    verdict = validate_call(
        "add_task", {"title": "x", "type": "errand", "duration_min": 30}, state, base
    )
    assert isinstance(verdict, Accepted)
    assert isinstance(verdict.args, AddTaskArgs)
    assert verdict.args.type is TaskType.ERRAND
    assert verdict.args.duration_min == 30


# --- allowlist and shape ---------------------------------------------------------------------


def test_unknown_tool(codes: Codes) -> None:
    assert codes("send_email", {}) == [ErrorCode.UNKNOWN_TOOL]


def test_non_string_tool_name(codes: Codes) -> None:
    assert codes(None, {}) == [ErrorCode.UNKNOWN_TOOL]


def test_args_not_an_object(codes: Codes) -> None:
    assert codes("plan_day", "2026-10-29") == [ErrorCode.BAD_TYPE]


def test_unexpected_arg(codes: Codes) -> None:
    assert codes("recommend_now", {"now": "2026-10-28T16:00"}) == [ErrorCode.UNEXPECTED_ARG]


def test_model_cannot_set_task_id_on_add(codes: Codes) -> None:
    assert ErrorCode.UNEXPECTED_ARG in codes("add_task", {"title": "x", "id": "T9"})


def test_missing_required_arg(codes: Codes) -> None:
    assert codes("travel_time", {"from_place": "home", "mode": "drive"}) == [ErrorCode.MISSING_ARG]


@pytest.mark.parametrize("duration", [4, 721, 0, -5])
def test_duration_out_of_range(duration: int, codes: Codes) -> None:
    assert codes("add_task", {"title": "x", "duration_min": duration}) == [ErrorCode.OUT_OF_RANGE]


@pytest.mark.parametrize("duration", [5, 720])
def test_duration_bounds_inclusive(duration: int, codes: Codes) -> None:
    assert codes("add_task", {"title": "x", "duration_min": duration}) == []


@pytest.mark.parametrize("duration", ["20", 20.5, True])
def test_duration_strictly_typed(duration: object, codes: Codes) -> None:
    assert codes("add_task", {"title": "x", "duration_min": duration}) == [ErrorCode.BAD_TYPE]


def test_empty_title(codes: Codes) -> None:
    assert codes("add_task", {"title": ""}) == [ErrorCode.BAD_VALUE]


def test_mode_must_be_drive_or_transit(codes: Codes) -> None:
    args = {"from_place": "home", "to_place": "um", "mode": "walk"}
    assert codes("travel_time", args) == [ErrorCode.OUT_OF_RANGE]


def test_bad_task_type(codes: Codes) -> None:
    assert codes("add_task", {"title": "x", "type": "meeting"}) == [ErrorCode.OUT_OF_RANGE]


def test_prerequisites_wrong_type(codes: Codes) -> None:
    assert codes("add_task", {"title": "x", "prerequisites": "T2"}) == [ErrorCode.BAD_TYPE]


# --- datetimes and horizon -------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["tomorrow", "2026-10-30", "17:15", "2026-13-40T10:00"])
def test_bad_datetime(bad: object, codes: Codes) -> None:
    assert codes("add_task", {"title": "x", "deadline": bad}) == [ErrorCode.BAD_DATETIME]


def test_timezone_offset_rejected(codes: Codes) -> None:
    args = {"title": "x", "deadline": "2026-10-30T10:00+08:00"}
    assert codes("add_task", args) == [ErrorCode.BAD_DATETIME]


def test_datetime_in_past(codes: Codes) -> None:
    args = {"title": "x", "deadline": "2026-10-28T15:39"}
    assert codes("add_task", args) == [ErrorCode.OUT_OF_HORIZON]


def test_datetime_beyond_horizon(codes: Codes) -> None:
    args = {"title": "x", "deadline": "2026-11-04T15:41"}
    assert codes("add_task", args) == [ErrorCode.OUT_OF_HORIZON]


@pytest.mark.parametrize("edge", ["2026-10-28T15:40", "2026-11-04T15:40"])
def test_horizon_edges_inclusive(edge: str, codes: Codes) -> None:
    assert codes("add_task", {"title": "x", "deadline": edge}) == []


@pytest.mark.parametrize("day", ["2026-10-27", "2026-11-12"])
def test_plan_day_outside_horizon(day: str, codes: Codes) -> None:
    assert codes("plan_day", {"date": day}) == [ErrorCode.OUT_OF_HORIZON]


@pytest.mark.parametrize("bad", ["Thursday", 20261029, "2026-10-29T10:00:00"])
def test_plan_day_bad_date(bad: object, codes: Codes) -> None:
    assert codes("plan_day", {"date": bad}) == [ErrorCode.BAD_DATETIME]


# --- places and ids --------------------------------------------------------------------------


def test_unknown_place_is_rejected(codes: Codes) -> None:
    """A haircut at an unknown salon must be rejected, not invented."""
    args = {"title": "Haircut", "place": "salon_bangsar"}
    assert codes("add_task", args) == [ErrorCode.UNKNOWN_PLACE]


def test_anywhere_is_a_valid_task_place(codes: Codes) -> None:
    assert codes("add_task", {"title": "x", "place": "anywhere"}) == []


def test_anywhere_is_not_a_travel_endpoint(codes: Codes) -> None:
    args = {"from_place": "anywhere", "to_place": "um", "mode": "drive"}
    assert codes("travel_time", args) == [ErrorCode.UNKNOWN_PLACE]


def test_feasibility_origin_must_be_known(codes: Codes) -> None:
    args = {"task_id": "T2", "start": "2026-10-28T17:15", "from_place": "mars"}
    assert codes("check_feasibility", args) == [ErrorCode.UNKNOWN_PLACE]


def test_unknown_task_on_delete(codes: Codes) -> None:
    assert codes("delete_task", {"task_id": "T99"}) == [ErrorCode.UNKNOWN_TASK]


def test_unknown_task_on_feasibility(codes: Codes) -> None:
    args = {"task_id": "T99", "start": "2026-10-28T17:15", "from_place": "office"}
    assert codes("check_feasibility", args) == [ErrorCode.UNKNOWN_TASK]


def test_unknown_prerequisite(codes: Codes) -> None:
    assert codes("add_task", {"title": "x", "prerequisites": ["T99"]}) == [ErrorCode.UNKNOWN_TASK]


# --- update_task -----------------------------------------------------------------------------


def test_update_cannot_change_id(codes: Codes) -> None:
    args = {"task_id": "T4", "fields": {"id": "T9"}}
    assert codes("update_task", args) == [ErrorCode.UNEXPECTED_ARG]


def test_unexpected_args_name_the_valid_ones(state: SessionState, base: BaseData) -> None:
    verdict = validate_call("update_task", {"task_id": "T4", "fields": {"done": True}}, state, base)
    assert isinstance(verdict, Rejected)
    assert "status" in verdict.message()
    assert verdict.issues[0].field == "fields.done"
    top = validate_call("recommend_now", {"now": "x"}, state, base)
    assert isinstance(top, Rejected)
    assert "takes no arguments" in top.message()


def test_update_fields_are_validated(codes: Codes) -> None:
    args = {"task_id": "T4", "fields": {"duration_min": 9999}}
    assert codes("update_task", args) == [ErrorCode.OUT_OF_RANGE]


def test_update_with_no_fields(codes: Codes) -> None:
    assert codes("update_task", {"task_id": "T4", "fields": {}}) == [ErrorCode.BAD_VALUE]


def test_update_missing_fields(codes: Codes) -> None:
    assert codes("update_task", {"task_id": "T4"}) == [ErrorCode.MISSING_ARG]


def test_update_with_only_nulls_is_empty(codes: Codes) -> None:
    args = {"task_id": "T4", "fields": {"title": None}}
    assert codes("update_task", args) == [ErrorCode.BAD_VALUE]


def test_update_checks_place_and_horizon(codes: Codes) -> None:
    args = {"task_id": "T4", "fields": {"place": "mars", "deadline": "2027-01-01T00:00"}}
    assert codes("update_task", args) == [ErrorCode.OUT_OF_HORIZON, ErrorCode.UNKNOWN_PLACE]


def test_update_self_prerequisite_is_a_cycle(codes: Codes) -> None:
    args = {"task_id": "T4", "fields": {"prerequisites": ["T4"]}}
    assert codes("update_task", args) == [ErrorCode.CYCLE]


def test_update_transitive_cycle(codes: Codes) -> None:
    # T5 already depends on T2, so making T2 depend on T5 closes the loop.
    args = {"task_id": "T2", "fields": {"prerequisites": ["T5"]}}
    assert codes("update_task", args) == [ErrorCode.CYCLE]


def test_update_acyclic_prerequisite(codes: Codes) -> None:
    args = {"task_id": "T3", "fields": {"prerequisites": ["T2"]}}
    assert codes("update_task", args) == []


def test_update_unknown_prerequisite_is_not_reported_as_cycle(codes: Codes) -> None:
    args = {"task_id": "T3", "fields": {"prerequisites": ["T99"]}}
    assert codes("update_task", args) == [ErrorCode.UNKNOWN_TASK]


# --- malformed calls -------------------------------------------------------------------------


def test_malformed_call_is_rejected_visibly() -> None:
    verdict = reject_malformed("add_task", "invalid JSON")
    assert isinstance(verdict, Rejected)
    assert verdict.issues[0].code is ErrorCode.MALFORMED_CALL
    assert "invalid JSON" in verdict.message()


def test_malformed_call_without_details() -> None:
    assert reject_malformed(None, "").issues[0].field == "name"


# --- cycle detection against an independent oracle -------------------------------------------

_IDS = [f"N{i}" for i in range(6)]


def _state(edges: dict[str, tuple[str, ...]]) -> SessionState:
    tasks = tuple(
        Task(id=i, title=i, type=TaskType.FLEXIBLE, prerequisites=edges.get(i, ())) for i in _IDS
    )
    return SessionState(now=datetime(2026, 10, 28, 15, 40), location="home", tasks=tasks)


def _reaches(edges: dict[str, tuple[str, ...]], start: str, goal: str) -> bool:
    """Oracle: transitive closure by fixed-point iteration (independent of the DFS)."""
    reach = {start}
    changed = True
    while changed:
        changed = False
        for node in list(reach):  # noqa: PERF101 - set is mutated inside the loop
            for nxt in edges.get(node, ()):
                if nxt not in reach:
                    reach.add(nxt)
                    changed = True
    return goal in reach


@given(
    edges=st.dictionaries(
        st.sampled_from(_IDS), st.lists(st.sampled_from(_IDS), max_size=3, unique=True).map(tuple)
    ),
    task_id=st.sampled_from(_IDS),
    new=st.lists(st.sampled_from(_IDS), max_size=3, unique=True).map(tuple),
)
def test_creates_cycle_matches_oracle(
    edges: dict[str, tuple[str, ...]], task_id: str, new: tuple[str, ...]
) -> None:
    edges = {k: v for k, v in edges.items() if k != task_id}  # task_id's old edges are replaced
    # A valid state is already acyclic; only the proposed edit may introduce a cycle.
    assume(not any(_reaches(edges, p, n) for n, ps in edges.items() for p in ps))
    expected = any(p == task_id or _reaches(edges, p, task_id) for p in new)
    assert creates_cycle(_state(edges), task_id, new) is expected
