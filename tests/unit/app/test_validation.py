# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
from datetime import datetime
from typing import Any

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from planner.app.prompt import forwarded
from planner.app.tools.args import (
    AddTaskArgs,
    DeleteTaskArgs,
    PlanDayArgs,
    RecommendNowArgs,
    SetLocationArgs,
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
from planner.domain.errors import ErrorCode, Issue
from planner.domain.models import BaseData, SessionState, Task, TaskType
from tests.conftest import SAID, Codes

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
        ("set_location", {"place_id": "library"}, SetLocationArgs),
        ("delete_task", {"task_id": "T4"}, DeleteTaskArgs),
        ("delete_task", {"task_id": "all"}, DeleteTaskArgs),
        ("update_task", {"task_id": "T4", "fields": {"status": "done"}}, UpdateTaskArgs),
        (
            "add_task",
            {
                "title": "Collect parcel",
                "duration_min": 45,
                "duration_quote": "45 minutes",
                "place": "post_office",
                "deadline": "2026-10-30T17:00",
                "deadline_quote": "by Friday",
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
    verdict = validate_call(name, args, state, base, said=(SAID,))
    assert isinstance(verdict, Accepted), verdict
    if expected_type is not None:
        assert isinstance(verdict.args, expected_type)


def test_accepted_args_are_typed(state: SessionState, base: BaseData) -> None:
    args = {"title": "x", "type": "errand", "duration_min": 45, "duration_quote": "45 minutes"}
    verdict = validate_call("add_task", args, state, base, said=(SAID,))
    assert isinstance(verdict, Accepted)
    assert isinstance(verdict.args, AddTaskArgs)
    assert verdict.args.type is TaskType.ERRAND
    assert verdict.args.duration_min == 45


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


@pytest.mark.parametrize(("duration", "quote"), [(5, "5 minutes"), (720, "12 hours")])
def test_duration_bounds_inclusive(
    duration: int, quote: str, state: SessionState, base: BaseData
) -> None:
    args = {"title": "x", "duration_min": duration, "duration_quote": quote}
    assert _issues("add_task", args, (f"Add it, {quote}",), state, base) == []


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


@pytest.mark.parametrize("bad", ["tomorrow", "17:15", "2026-13-40T10:00", "2026-13-40"])
def test_bad_datetime(bad: object, codes: Codes) -> None:
    assert codes("add_task", {"title": "x", "deadline": bad}) == [ErrorCode.BAD_DATETIME]


def test_timezone_offset_rejected(codes: Codes) -> None:
    args = {"title": "x", "deadline": "2026-10-30T10:00+08:00"}
    assert codes("add_task", args) == [ErrorCode.BAD_DATETIME]


def test_datetime_in_past(codes: Codes) -> None:
    args = {"title": "x", "deadline": "2026-10-28T15:39", "deadline_quote": "by Friday"}
    assert codes("add_task", args) == [ErrorCode.OUT_OF_HORIZON]


def test_datetime_beyond_horizon(codes: Codes) -> None:
    args = {"title": "x", "deadline": "2026-11-04T15:41", "deadline_quote": "by Friday"}
    assert codes("add_task", args) == [ErrorCode.OUT_OF_HORIZON]


@pytest.mark.parametrize("edge", ["2026-10-28T15:40", "2026-11-04T15:40"])
def test_horizon_edges_inclusive(edge: str, codes: Codes) -> None:
    args = {"title": "x", "deadline": edge, "deadline_quote": "by Friday"}
    assert codes("add_task", args) == []


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


def test_set_location_needs_a_known_place(codes: Codes) -> None:
    assert codes("set_location", {"place_id": "mars"}) == [ErrorCode.UNKNOWN_PLACE]
    assert codes("set_location", {"place_id": "anywhere"}) == [ErrorCode.UNKNOWN_PLACE]


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


def test_unexpected_args_at_one_level_list_the_valid_ones_once(
    state: SessionState, base: BaseData
) -> None:
    args = {"task_id": "T4", "fields": {"done": True, "priority": 1}}
    verdict = validate_call("update_task", args, state, base)
    assert isinstance(verdict, Rejected)
    assert len(verdict.issues) == 1
    assert verdict.issues[0].field == "fields.done, fields.priority"
    assert verdict.message().count("valid:") == 1


def test_unexpected_args_at_two_levels_give_one_issue_each(
    state: SessionState, base: BaseData
) -> None:
    args = {"task_id": "T4", "id": "T9", "fields": {"done": True}}
    verdict = validate_call("update_task", args, state, base)
    assert isinstance(verdict, Rejected)
    assert sorted(i.field for i in verdict.issues) == ["fields.done", "id"]


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
    fields = {"place": "mars", "deadline": "2027-01-01T00:00", "deadline_quote": "by Friday"}
    args = {"task_id": "T4", "fields": fields}
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


def test_fixed_event_needs_a_start_time(codes: Codes) -> None:
    args = {"title": "Exam", "type": "fixed_event"}
    assert codes("add_task", args) == [ErrorCode.MISSING_ARG]
    assert codes("add_task", {**args, "earliest_start": "2026-10-29T09:00"}) == []


def test_deadline_before_earliest_start_is_rejected_on_add(codes: Codes) -> None:
    args = {
        "title": "x",
        "earliest_start": "2026-10-29T12:00",
        "deadline": "2026-10-29T10:00",
        "deadline_quote": "by Friday",
    }
    assert codes("add_task", args) == [ErrorCode.BAD_VALUE]


def test_update_cannot_turn_a_task_into_a_fixed_event_without_a_start(codes: Codes) -> None:
    args = {"task_id": "T3", "fields": {"type": "fixed_event"}}
    assert codes("update_task", args) == [ErrorCode.MISSING_ARG]


def test_update_checks_the_merged_task(codes: Codes) -> None:
    # T3 has no earliest_start; giving one after its deadline is a contradiction.
    args = {"task_id": "T3", "fields": {"earliest_start": "2030-01-01T00:00"}}
    assert ErrorCode.OUT_OF_HORIZON in codes("update_task", args)
    late = {"task_id": "T3", "fields": {"earliest_start": "2026-11-03T09:00"}}
    assert codes("update_task", late) == [ErrorCode.BAD_VALUE]


# --- stated values: a duration or deadline must quote the user -------------------------------

HAIRCUT = "Add a haircut at the salon in Bangsar this week"


def _issues(
    name: str, args: dict[str, Any], said: tuple[str, ...], state: SessionState, base: BaseData
) -> list[Issue]:
    verdict = validate_call(name, args, state, base, said)
    return [] if isinstance(verdict, Accepted) else list(verdict.issues)


def test_invented_duration_without_a_quote_is_rejected(state: SessionState, base: BaseData) -> None:
    [issue] = _issues("add_task", {"title": "Haircut", "duration_min": 60}, (HAIRCUT,), state, base)
    assert issue.code is ErrorCode.UNSTATED_VALUE
    assert issue.field == "duration_quote"
    assert "do not guess: ask the user for it, or leave duration_min out" in issue.message


def test_quote_the_user_did_not_say_is_rejected(state: SessionState, base: BaseData) -> None:
    args = {"title": "Haircut", "deadline": "2026-11-01T18:00", "deadline_quote": "by Sunday"}
    [issue] = _issues("add_task", args, (HAIRCUT,), state, base)
    assert (issue.code, issue.field) == (ErrorCode.UNSTATED_VALUE, "deadline_quote")
    assert issue.message.startswith(
        "'by Sunday' is not in the user's messages this turn. "
        "If the user stated the deadline, copy their exact words. "
        "If the user has not stated a deadline, do not guess"
    )


def test_stated_values_are_accepted_ignoring_case_and_spacing(
    state: SessionState, base: BaseData
) -> None:
    args = {
        "title": "Haircut",
        "duration_min": 60,
        "duration_quote": "ABOUT an hour",
        "deadline": "2026-11-01",
        "deadline_quote": "this  week",
    }
    said = ("Add a haircut,\nabout   an hour, this\tweek",)
    assert _issues("add_task", args, said, state, base) == []


@pytest.mark.parametrize("quote", ["salon", "a", "this"])
def test_a_duration_quote_must_state_an_amount_of_time(
    quote: str, state: SessionState, base: BaseData
) -> None:
    args = {"title": "Haircut", "duration_min": 60, "duration_quote": quote}
    [issue] = _issues("add_task", args, (HAIRCUT,), state, base)
    assert (issue.code, issue.field) == (ErrorCode.UNSTATED_VALUE, "duration_quote")
    assert f"'{quote}' does not state a duration" in issue.message


@pytest.mark.parametrize(
    ("quote", "minutes"),
    [("about 20 minutes", 20), ("2 hours", 120), ("an hour", 60), ("1h", 60), ("Half an Hour", 30)],
)
def test_a_duration_quote_with_an_amount_of_time_is_accepted(
    quote: str, minutes: int, state: SessionState, base: BaseData
) -> None:
    args = {"title": "Haircut", "duration_min": minutes, "duration_quote": quote}
    assert _issues("add_task", args, (f"Add a haircut, {quote}",), state, base) == []


@pytest.mark.parametrize(
    ("quote", "minutes"),
    [("1.5 hours", 90), ("three hrs", 180), ("half hour", 30), ("45 mins", 45), ("12h", 720)],
)
def test_a_duration_must_be_the_amount_its_quote_states(
    quote: str, minutes: int, state: SessionState, base: BaseData
) -> None:
    said = (f"Study, {quote}",)
    args = {"title": "Study", "duration_min": minutes, "duration_quote": quote}
    assert _issues("add_task", args, said, state, base) == []
    wrong = {**args, "duration_min": minutes // 2}
    [issue] = _issues("add_task", wrong, said, state, base)
    assert (issue.code, issue.field) == (ErrorCode.UNSTATED_VALUE, "duration_min")
    assert issue.message == f"'{quote}' is {minutes} minutes, not {minutes // 2}: use {minutes}"


def test_a_duration_within_a_tenth_of_its_quote_is_accepted(
    state: SessionState, base: BaseData
) -> None:
    said = ("Study for 2 hours",)
    near = {"title": "Study", "duration_min": 108, "duration_quote": "2 hours"}
    assert _issues("add_task", near, said, state, base) == []
    assert _issues("add_task", {**near, "duration_min": 107}, said, state, base) != []


def test_an_updated_duration_must_be_the_amount_its_quote_states(
    state: SessionState, base: BaseData
) -> None:
    args = {"task_id": "T3", "fields": {"duration_min": 60, "duration_quote": "30 min"}}
    [issue] = _issues("update_task", args, ("make it 30 min",), state, base)
    assert (issue.code, issue.field) == (ErrorCode.UNSTATED_VALUE, "fields.duration_min")


@pytest.mark.parametrize(
    ("quote", "minutes"),
    [
        ("an hour and a half", 90),
        ("1-2 hours", 90),
        ("2 hours 30 minutes", 150),
        ("twenty five minutes", 25),
    ],
)
def test_an_amount_that_is_not_plain_is_not_checked(
    quote: str, minutes: int, state: SessionState, base: BaseData
) -> None:
    args = {"title": "Study", "duration_min": minutes, "duration_quote": quote}
    assert _issues("add_task", args, (f"Study, {quote}",), state, base) == []


def test_a_duration_too_long_for_one_task_says_how_to_go_on(
    state: SessionState, base: BaseData
) -> None:
    said = ("Schedule 14 hours of assignment work on Friday",)
    args = {"title": "Assignment", "duration_min": 1440, "duration_quote": "14 hours"}
    [issue] = _issues("add_task", args, said, state, base)
    assert (issue.code, issue.field) == (ErrorCode.OUT_OF_RANGE, "duration_min")
    assert issue.message.startswith(
        "1440 is more than 720 minutes (12 hours), the most for one task. "
        "'14 hours' is 840 minutes: add it as 2 tasks of 420 minutes, each with duration_quote "
        "'14 hours', then plan the day with plan_day. "
        "The daily limit is the user's own setting, not a reason to refuse"
    )
    part = {**args, "duration_min": 420}
    assert _issues("add_task", part, said, state, base) == []
    [wrong] = _issues("add_task", {**args, "duration_min": 14}, said, state, base)
    assert (wrong.code, wrong.field) == (ErrorCode.UNSTATED_VALUE, "duration_min")
    assert wrong.message.startswith(
        "'14 hours' is 840 minutes, not 14: add it as 2 tasks of 420 minutes, each with "
        "duration_quote '14 hours', then plan the day with plan_day."
    )


def test_a_too_long_duration_with_a_quote_that_fits_gives_the_number(
    state: SessionState, base: BaseData
) -> None:
    args = {"task_id": "T1", "fields": {"duration_min": 900, "duration_quote": "12 hours"}}
    [issue] = _issues("update_task", args, ("make it 12 hours",), state, base)
    assert (issue.code, issue.field) == (ErrorCode.OUT_OF_RANGE, "fields.duration_min")
    assert issue.message.endswith("the most for one task. '12 hours' is 720 minutes: use 720")


@pytest.mark.parametrize("quote", [None, "all day"])
def test_a_too_long_duration_without_an_amount_says_how_to_go_on(
    quote: str | None, state: SessionState, base: BaseData
) -> None:
    args = {"title": "Study", "duration_min": 800, "duration_quote": quote}
    [issue] = _issues("add_task", args, ("Study all day",), state, base)
    assert issue.message.startswith(
        "800 is more than 720 minutes (12 hours), the most for one task. Add longer work"
    )


def test_a_deadline_quote_must_be_more_than_junk(state: SessionState, base: BaseData) -> None:
    args = {"title": "Haircut", "deadline": "2026-11-01T18:00", "deadline_quote": "a"}
    [issue] = _issues("add_task", args, (HAIRCUT,), state, base)
    assert (issue.code, issue.field) == (ErrorCode.UNSTATED_VALUE, "deadline_quote")
    assert "'a' does not state a deadline" in issue.message
    punctuation = {**args, "deadline_quote": "..."}
    assert _issues("add_task", punctuation, ("Add it ...",), state, base) != []
    week = {**args, "deadline_quote": "this week"}
    assert _issues("add_task", week, (HAIRCUT,), state, base) == []


def test_a_blank_quote_counts_as_missing(state: SessionState, base: BaseData) -> None:
    args = {"title": "x", "duration_min": 30, "duration_quote": "  "}
    [issue] = _issues("add_task", args, (HAIRCUT,), state, base)
    assert "duration_min needs duration_quote" in issue.message


def test_any_message_of_the_turn_can_state_the_value(state: SessionState, base: BaseData) -> None:
    args = {"title": "x", "duration_min": 30, "duration_quote": "half an hour"}
    assert _issues("add_task", args, (HAIRCUT, "half an hour"), state, base) == []


def test_pasted_text_is_not_the_users_own_words(state: SessionState, base: BaseData) -> None:
    args = {"title": "Reply", "deadline": "2026-10-30T17:00", "deadline_quote": "by Friday"}
    said = (forwarded("Please reply by Friday."),)
    [issue] = _issues("add_task", args, said, state, base)
    assert issue.code is ErrorCode.UNSTATED_VALUE


def test_update_without_a_duration_or_deadline_needs_no_quote(
    state: SessionState, base: BaseData
) -> None:
    for fields in ({"status": "done"}, {"title": "Renamed"}, {"place": "home"}):
        assert _issues("update_task", {"task_id": "T3", "fields": fields}, (), state, base) == []


def test_update_of_a_duration_needs_the_users_words(state: SessionState, base: BaseData) -> None:
    args = {"task_id": "T3", "fields": {"duration_min": 30}}
    [issue] = _issues("update_task", args, ("make it longer",), state, base)
    assert (issue.code, issue.field) == (ErrorCode.UNSTATED_VALUE, "fields.duration_quote")
    stated = {"task_id": "T3", "fields": {"duration_min": 30, "duration_quote": "30 min"}}
    assert _issues("update_task", stated, ("make it 30 min",), state, base) == []


def test_a_quote_alone_is_not_a_change(codes: Codes) -> None:
    args = {"task_id": "T3", "fields": {"duration_quote": "45 minutes"}}
    assert codes("update_task", args) == [ErrorCode.BAD_VALUE]


def test_resending_the_stored_value_needs_no_quote(state: SessionState, base: BaseData) -> None:
    said = ("Schedule 14 hours of assignment work on Friday",)
    fields = {
        "title": "Finish assignment part 1",
        "duration_min": 360,
        "deadline": "2026-11-01T23:59:00Z",
        "deadline_quote": "by Sun 11-01 23:59",
    }
    assert _issues("update_task", {"task_id": "T1", "fields": fields}, said, state, base) == []
    moved = {**fields, "deadline": "2026-10-31T23:59"}
    [issue] = _issues("update_task", {"task_id": "T1", "fields": moved}, said, state, base)
    assert issue.field == "fields.deadline_quote"


# --- datetime formats the model writes -------------------------------------------------------


def test_date_only_deadline_means_the_end_of_that_day(state: SessionState, base: BaseData) -> None:
    args = {"title": "x", "deadline": "2026-10-30", "deadline_quote": "by Friday"}
    verdict = validate_call("add_task", args, state, base, (SAID,))
    assert isinstance(verdict, Accepted)
    assert isinstance(verdict.args, AddTaskArgs)
    assert verdict.args.deadline == datetime(2026, 10, 30, 23, 59)


@pytest.mark.parametrize("value", ["2026-10-30T17:00Z", "2026-10-30T17:00:00z"])
def test_a_z_suffix_is_read_as_the_session_clock(
    value: str, state: SessionState, base: BaseData
) -> None:
    args = {"title": "x", "deadline": value, "deadline_quote": "by Friday", "earliest_start": value}
    verdict = validate_call("add_task", args, state, base, (SAID,))
    assert isinstance(verdict, Accepted)
    assert isinstance(verdict.args, AddTaskArgs)
    assert verdict.args.deadline == verdict.args.earliest_start == datetime(2026, 10, 30, 17, 0)


def test_date_only_with_z_is_still_a_date(state: SessionState, base: BaseData) -> None:
    args = {"title": "x", "deadline": "2026-10-30Z", "deadline_quote": "by Friday"}
    verdict = validate_call("add_task", args, state, base, (SAID,))
    assert isinstance(verdict, Accepted)
    assert isinstance(verdict.args, AddTaskArgs)
    assert verdict.args.deadline == datetime(2026, 10, 30, 23, 59)


def test_date_only_start_is_still_rejected(codes: Codes) -> None:
    assert codes("add_task", {"title": "x", "earliest_start": "2026-10-30"}) == [
        ErrorCode.BAD_DATETIME
    ]
