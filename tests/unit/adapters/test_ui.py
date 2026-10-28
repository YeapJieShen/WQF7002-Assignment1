from collections.abc import Iterator, Sequence
from typing import Any

import gradio as gr
import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, AnyMessage

from planner.adapters.session import Session
from planner.adapters.ui import (
    BACKEND_DOWN,
    answer,
    build_app,
    default_session,
    send,
    start,
    task_rows,
)
from planner.domain.models import BaseData, SessionState


class ScriptedModel:
    def __init__(self, *script: AIMessage) -> None:
        self._script: Iterator[AIMessage] = iter(script)

    def invoke(self, messages: Sequence[AnyMessage], /) -> AIMessage:  # noqa: ARG002 - protocol
        return next(self._script)


class DownModel:
    def invoke(self, messages: Sequence[AnyMessage], /) -> AIMessage:  # noqa: ARG002 - protocol
        request: Any = httpx.Request("POST", "http://down")
        raise openai.APIConnectionError(request=request)


def call(name: str, args: dict[str, object]) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": "c1"}])


def make(model: object, state: SessionState, base: BaseData) -> Session:
    return Session(model, base, "rules", state)  # type: ignore[arg-type]


def test_task_rows_describe_each_task(state: SessionState) -> None:
    rows = task_rows(state)
    assert [r[0] for r in rows] == [t.id for t in state.tasks]
    assert all(len(r) == 7 for r in rows)


def test_start_shows_the_starting_tasks(state: SessionState, base: BaseData) -> None:
    session, tasks, log = start(lambda: make(ScriptedModel(), state, base))
    assert tasks == task_rows(state)
    assert log == []
    assert session.state == state


def test_send_appends_both_sides_and_refreshes_the_tables(
    state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(call("add_task", {"title": "Post parcel"}), AIMessage("Added."))
    session = make(model, state, base)
    history, box, tasks, log, card, question = send("add a parcel task", [], session)
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[-1]["content"] == "Added."
    assert any(row[1] == "Post parcel" for row in tasks)
    assert [row[0] for row in log] == ["check", "act"]
    assert box["interactive"] is True
    assert card["visible"] is False
    assert question == ""


def test_blank_message_changes_nothing(state: SessionState, base: BaseData) -> None:
    out = send("  ", [], make(ScriptedModel(), state, base))
    assert len(out) == 6


def test_delete_shows_the_card_then_the_answer_settles_it(
    state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(call("delete_task", {"task_id": "T4"}), AIMessage("Deleted."))
    session = make(model, state, base)
    history, box, _, _, card, question = send("delete T4", [], session)
    assert card["visible"] is True
    assert "T4" in question
    assert box["interactive"] is False
    history, box, tasks, _, card, _ = answer(approved=True, history=history, session=session)
    assert history[-1]["content"] == "Deleted."
    assert all(row[0] != "T4" for row in tasks)
    assert card["visible"] is False
    assert box["interactive"] is True


def test_backend_failure_becomes_a_visible_error(state: SessionState, base: BaseData) -> None:
    with pytest.raises(gr.Error, match=BACKEND_DOWN):
        send("hi", [], make(DownModel(), state, base))


def test_build_app_lays_out_without_launching(state: SessionState, base: BaseData) -> None:
    assert isinstance(build_app(lambda: make(ScriptedModel(), state, base)), gr.Blocks)


def test_default_session_builds_without_calling_a_model() -> None:
    assert isinstance(default_session(), Session)


def test_a_lost_session_asks_for_a_reload() -> None:
    with pytest.raises(gr.Error, match="Reload the page"):
        send("hi", [], None)
    with pytest.raises(gr.Error, match="Reload the page"):
        answer(approved=True, history=[], session=None)
