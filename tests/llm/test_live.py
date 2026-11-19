# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
"""Needs a running backend (see PLANNER_LLM_ENDPOINTS). Never run in CI: `pytest -m llm`."""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from planner.adapters.config import Settings
from planner.adapters.llm import build_model

pytestmark = pytest.mark.llm


def test_model_picks_the_right_tool_for_a_simple_request() -> None:
    model = build_model(Settings())
    reply = model.invoke(
        [
            SystemMessage("You manage the user's task list. Use the tools; do not guess."),
            HumanMessage("Add a task to post a parcel, 20 minutes."),
        ]
    )
    assert [c["name"] for c in reply.tool_calls] == ["add_task"]
    assert reply.tool_calls[0]["args"]["title"]
