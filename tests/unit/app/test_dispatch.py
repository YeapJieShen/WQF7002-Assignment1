import pytest

from planner.app.tools.args import TOOLS_BY_NAME
from planner.app.tools.dispatch import run_tool
from planner.app.validation import Accepted, validate_call
from planner.domain.models import BaseData, SessionState

CALLS = {
    "add_task": {"title": "Post parcel", "duration_min": 20, "place": "post_office"},
    "update_task": {"task_id": "T3", "fields": {"duration_min": 30}},
    "delete_task": {"task_id": "T4"},
    "travel_time": {"from_place": "office", "to_place": "um", "mode": "drive"},
    "check_feasibility": {"task_id": "T3", "start": "2026-10-28T15:40", "from_place": "office"},
    "recommend_now": {},
    "plan_day": {"date": "2026-10-29"},
}
EDITING = {"add_task", "update_task", "delete_task"}


def test_every_registered_tool_has_an_example() -> None:
    assert set(CALLS) == set(TOOLS_BY_NAME)


@pytest.mark.parametrize("name", sorted(CALLS))
def test_validated_call_runs_and_only_editing_tools_change_state(
    name: str, state: SessionState, base: BaseData
) -> None:
    verdict = validate_call(name, CALLS[name], state, base)
    assert isinstance(verdict, Accepted)
    result = run_tool(verdict.args, state, base)
    assert result.message
    assert (result.state != state) is (name in EDITING)
