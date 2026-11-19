# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
import pytest

from planner.app.tools.args import TOOLS_BY_NAME
from planner.app.tools.dispatch import run_tool
from planner.app.validation import Accepted, validate_call
from planner.domain.models import BaseData, SessionState
from tests.conftest import SAID

CALLS = {
    "add_task": {
        "title": "Post parcel",
        "duration_min": 45,
        "duration_quote": "45 minutes",
        "place": "post_office",
    },
    "update_task": {
        "task_id": "T3",
        "fields": {"deadline": "2026-10-30T17:00", "deadline_quote": "by Friday"},
    },
    "delete_task": {"task_id": "T4"},
    "travel_time": {"from_place": "office", "to_place": "um", "mode": "drive"},
    "check_feasibility": {"task_id": "T3", "start": "2026-10-28T15:40", "from_place": "office"},
    "recommend_now": {},
    "plan_day": {"date": "2026-10-29"},
    "set_location": {"place_id": "home"},
}
EDITING = {"add_task", "update_task", "delete_task", "set_location"}


def test_every_registered_tool_has_an_example() -> None:
    assert set(CALLS) == set(TOOLS_BY_NAME)


@pytest.mark.parametrize("name", sorted(CALLS))
def test_validated_call_runs_and_only_editing_tools_change_state(
    name: str, state: SessionState, base: BaseData
) -> None:
    verdict = validate_call(name, CALLS[name], state, base, said=(SAID,))
    assert isinstance(verdict, Accepted)
    result = run_tool(verdict.args, state, base)
    assert result.message
    assert (result.state != state) is (name in EDITING)
