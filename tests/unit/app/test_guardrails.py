import pytest

from planner.app.guardrails import confirmation_question
from planner.app.tools.args import TOOLS_BY_NAME
from planner.app.validation import Accepted, validate_call
from planner.domain.models import BaseData, SessionState

CALLS: dict[str, dict[str, object]] = {
    "add_task": {"title": "Post parcel"},
    "update_task": {"task_id": "T3", "fields": {"duration_min": 30}},
    "delete_task": {"task_id": "T4"},
    "travel_time": {"from_place": "office", "to_place": "um", "mode": "drive"},
    "check_feasibility": {"task_id": "T3", "start": "2026-10-28T15:40", "from_place": "office"},
    "recommend_now": {},
    "plan_day": {"date": "2026-10-29"},
    "set_location": {"place_id": "home"},
}


def _question(name: str, raw: dict[str, object], state: SessionState, base: BaseData) -> str | None:
    verdict = validate_call(name, raw, state, base)
    assert isinstance(verdict, Accepted)
    return confirmation_question(verdict.args, state)


def test_every_registered_tool_is_classified() -> None:
    assert set(CALLS) == set(TOOLS_BY_NAME)


@pytest.mark.parametrize("name", sorted(set(CALLS) - {"delete_task"}))
def test_only_deletion_asks_first(name: str, state: SessionState, base: BaseData) -> None:
    assert _question(name, CALLS[name], state, base) is None


def test_deleting_one_task_names_it(state: SessionState, base: BaseData) -> None:
    question = _question("delete_task", CALLS["delete_task"], state, base)
    assert question == "Delete T4 Reply to supervisor? This cannot be undone."


def test_deleting_everything_states_how_many(state: SessionState, base: BaseData) -> None:
    question = _question("delete_task", {"task_id": "all"}, state, base)
    assert question == f"Delete all {len(state.tasks)} tasks? This cannot be undone."
