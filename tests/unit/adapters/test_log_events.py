"""One test per logged event family, plus the privacy rules. The model is always scripted."""

import logging
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from planner.adapters.graph import build_graph
from planner.domain.models import BaseData, SessionState
from tests.unit.adapters.fakes import ScriptedModel, call

CONFIG: Any = {"configurable": {"thread_id": "t"}}


@pytest.fixture(autouse=True)
def _capture(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="planner")


def fields(caplog: pytest.LogCaptureFixture, event: str) -> list[dict[str, Any]]:
    return [r.fields for r in caplog.records if r.getMessage() == event]  # type: ignore[attr-defined]  # fields is set by the logging adapter, not on LogRecord


def everything(caplog: pytest.LogCaptureFixture) -> str:
    """All text of every captured record: message, fields and any stack trace."""
    lines = [f"{r.getMessage()} {getattr(r, 'fields', {})}" for r in caplog.records]
    return " ".join(lines) + caplog.text


def run_graph(model: ScriptedModel, base: BaseData, state: SessionState, **kw: int) -> Any:
    graph = build_graph(model, base, "rules", **{"max_rejections": 3, **kw})
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


def test_graph_steps_are_timed_at_debug(
    caplog: pytest.LogCaptureFixture, state: SessionState, base: BaseData
) -> None:
    run_graph(ScriptedModel(AIMessage("Hello")), base, state)
    [step] = fields(caplog, "graph_step")
    assert step["node"] == "call_model"
    assert step["duration_ms"] >= 0
    assert caplog.records[0].levelno == logging.DEBUG


def test_an_accepted_call_logs_check_and_act(
    caplog: pytest.LogCaptureFixture, state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(call("add_task", {"title": "Post parcel"}), AIMessage("Added."))
    run_graph(model, base, state)
    [check] = fields(caplog, "tool_check")
    assert check == {"tool": "add_task", "accepted": True, "issue_codes": [], "rejections": 0}
    [act] = fields(caplog, "tool_act")
    assert (act["tool"], act["success"]) == ("add_task", True)
    assert act["duration_ms"] >= 0


def test_a_rejected_call_logs_only_issue_codes(
    caplog: pytest.LogCaptureFixture, state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(call("nope", {"title": "secret title"}), AIMessage("Sorry."))
    run_graph(model, base, state)
    [check] = fields(caplog, "tool_check")
    assert check["accepted"] is False
    assert check["issue_codes"] == ["unknown_tool"]
    assert check["rejections"] == 1


def test_confirmation_answers_are_logged(
    caplog: pytest.LogCaptureFixture, state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(call("delete_task", {"task_id": "T4"}), AIMessage("Kept."))
    graph, _ = run_graph(model, base, state)
    graph.invoke(Command(resume=False), CONFIG)
    assert fields(caplog, "confirm_answered") == [{"tool": "delete_task", "approved": False}]


def test_giving_up_is_logged(
    caplog: pytest.LogCaptureFixture, state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(
        *[
            call(
                "nope",
                {},
            )
            for _ in range(3)
        ]
    )
    run_graph(model, base, state, max_rejections=3)
    assert fields(caplog, "give_up") == [{"rejections": 3, "max_rejections": 3}]
    assert [f["rejections"] for f in fields(caplog, "tool_check")] == [1, 2, 3]


def test_a_failing_tool_is_logged_with_its_stack(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    state: SessionState,
    base: BaseData,
) -> None:
    def boom(*_: object) -> None:
        raise RuntimeError("tool broke")

    monkeypatch.setattr("planner.adapters.graph.run_tool", boom)
    model = ScriptedModel(call("add_task", {"title": "x"}))
    with pytest.raises(RuntimeError, match="tool broke"):
        run_graph(model, base, state)
    [act] = [r for r in caplog.records if r.getMessage() == "tool_act"]
    assert act.levelno == logging.ERROR
    assert act.exc_info is not None
    assert act.fields["success"] is False  # type: ignore[attr-defined]  # fields is set by the logging adapter, not on LogRecord


def test_tool_arguments_appear_only_at_debug(
    caplog: pytest.LogCaptureFixture, state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(call("add_task", {"title": "Hidden title"}), AIMessage("Added."))
    run_graph(model, base, state)
    [shown] = fields(caplog, "tool_args")
    assert "Hidden title" in shown["args"]
    caplog.clear()
    caplog.set_level(logging.INFO, logger="planner")
    run_graph(
        ScriptedModel(call("add_task", {"title": "Hidden title"}), AIMessage("Added.")),
        base,
        state,
    )
    assert "Hidden title" not in everything(caplog)
    assert not fields(caplog, "tool_args")
