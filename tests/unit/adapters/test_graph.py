import json
from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import Command

from planner.adapters.graph import DECLINED, GAVE_UP, build_graph
from planner.domain.models import BaseData, SessionState

CONFIG: Any = {"configurable": {"thread_id": "t"}}
RULES = "Be brief."


class ScriptedModel:
    """Replies with the scripted messages in order; fails the test if asked for more."""

    def __init__(self, *script: AIMessage) -> None:
        self._script: Iterator[AIMessage] = iter(script)
        self.calls = 0
        self.prompts: list[list[AnyMessage]] = []

    def invoke(self, messages: Sequence[AnyMessage], /) -> AIMessage:
        self.calls += 1
        self.prompts.append(list(messages))
        return next(self._script)


def call(name: str, args: dict[str, object], call_id: str = "c1") -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def start(model: ScriptedModel, base: BaseData, state: SessionState, **kwargs: int) -> Any:
    graph = build_graph(model, base, RULES, **{"max_rejections": 3, **kwargs})
    first = graph.invoke(
        {
            "messages": [HumanMessage("hi")],
            "session": state,
            "pending": None,
            "rejections": 0,
            "call_log": [],
        },
        CONFIG,
    )
    return graph, first


def tool_replies(result: Any) -> list[str]:
    return [str(m.content) for m in result["messages"] if isinstance(m, ToolMessage)]


def test_plain_reply_ends_the_turn(state: SessionState, base: BaseData) -> None:
    _, result = start(ScriptedModel(AIMessage("Hello")), base, state)
    assert result["messages"][-1].content == "Hello"
    assert result["session"] == state


def test_non_destructive_call_runs_without_asking(state: SessionState, base: BaseData) -> None:
    model = ScriptedModel(call("add_task", {"title": "Post parcel"}), AIMessage("Added."))
    _, result = start(model, base, state)
    assert "__interrupt__" not in result
    assert result["session"].task("T6") is not None
    assert "Added T6" in tool_replies(result)[0]


def test_delete_pauses_until_confirmed_then_deletes(state: SessionState, base: BaseData) -> None:
    model = ScriptedModel(call("delete_task", {"task_id": "T4"}), AIMessage("Deleted."))
    graph, first = start(model, base, state)
    assert "Delete T4 Reply to supervisor?" in first["__interrupt__"][0].value
    assert first["session"].task("T4") is not None
    final = graph.invoke(Command(resume=True), CONFIG)
    assert final["session"].task("T4") is None
    assert final["messages"][-1].content == "Deleted."


def test_declined_delete_changes_nothing(state: SessionState, base: BaseData) -> None:
    model = ScriptedModel(call("delete_task", {"task_id": "all"}), AIMessage("Kept them."))
    graph, _ = start(model, base, state)
    final = graph.invoke(Command(resume=False), CONFIG)
    assert final["session"] == state
    assert tool_replies(final) == [DECLINED]


def test_rejected_call_is_reported_and_can_be_corrected(
    state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(
        call("delete_everything", {}),
        call("add_task", {"title": "Post parcel"}),
        AIMessage("Added."),
    )
    _, result = start(model, base, state)
    replies = tool_replies(result)
    assert replies[0].startswith("Rejected:")
    assert "Added T6" in replies[1]
    assert result["rejections"] == 0


def test_malformed_call_is_rejected_visibly(state: SessionState, base: BaseData) -> None:
    bad = AIMessage(
        "",
        invalid_tool_calls=[
            {"name": "add_task", "args": "{oops", "id": "c1", "error": "bad json", "type": None}
        ],
    )
    _, result = start(ScriptedModel(bad, AIMessage("Sorry.")), base, state)
    assert tool_replies(result)[0].startswith("Rejected:")


def test_repeated_rejections_stop_at_the_cap(state: SessionState, base: BaseData) -> None:
    bad = [call("nope", {}, f"c{i}") for i in range(5)]
    model = ScriptedModel(*bad)
    _, result = start(model, base, state, max_rejections=3)
    assert model.calls == 3
    assert result["messages"][-1].content == GAVE_UP
    assert result["rejections"] == 0
    assert len(tool_replies(result)) == 3


@pytest.mark.parametrize("cap", [1, 2])
def test_cap_is_configurable(state: SessionState, base: BaseData, cap: int) -> None:
    model = ScriptedModel(*[call("nope", {}, f"c{i}") for i in range(5)])
    start(model, base, state, max_rejections=cap)
    assert model.calls == cap


def test_prompt_is_rules_then_history_then_fresh_snapshot(
    state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(call("add_task", {"title": "Post parcel"}), AIMessage("Added."))
    _, result = start(model, base, state)
    first, second = model.prompts
    assert isinstance(first[0], SystemMessage)
    assert first[0].content == RULES
    assert first[1].content == "hi"
    assert "Current session:" in str(first[-1].content)
    assert "Post parcel" not in str(first[-1].content)
    assert "Post parcel" in str(second[-1].content)
    assert not any(isinstance(m, SystemMessage) for m in result["messages"])
    assert not any("Current session:" in str(m.content) for m in result["messages"])


def test_call_log_records_an_accepted_call_through_to_its_result(
    state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(call("add_task", {"title": "Post parcel"}), AIMessage("Added."))
    _, result = start(model, base, state)
    check, act = result["call_log"]
    assert (check.stage, check.tool, check.outcome) == ("check", "add_task", "accepted")
    assert json.loads(check.args) == {"title": "Post parcel"}
    assert (act.stage, act.tool) == ("act", "add_task")
    assert "Added T6" in act.outcome


def test_call_log_records_rejections_including_malformed_calls(
    state: SessionState, base: BaseData
) -> None:
    bad = AIMessage(
        "",
        invalid_tool_calls=[
            {"name": "add_task", "args": "{oops", "id": "c1", "error": "bad json", "type": None}
        ],
    )
    model = ScriptedModel(call("nope", {}), bad, AIMessage("Sorry."))
    _, result = start(model, base, state)
    unknown, malformed = result["call_log"]
    assert unknown.outcome.startswith("rejected:")
    assert (malformed.tool, malformed.args) == ("add_task", "{oops")
    assert malformed.outcome.startswith("rejected:")


@pytest.mark.parametrize(("approved", "outcome"), [(True, "approved"), (False, "declined")])
def test_call_log_records_the_users_answer(
    state: SessionState, base: BaseData, *, approved: bool, outcome: str
) -> None:
    model = ScriptedModel(call("delete_task", {"task_id": "T4"}), AIMessage("Done."))
    graph, _ = start(model, base, state)
    final = graph.invoke(Command(resume=approved), CONFIG)
    stages = [(r.stage, r.outcome) for r in final["call_log"]]
    assert stages[:2] == [("check", "accepted"), ("confirm", outcome)]
    assert (len(stages) == 3) is approved
