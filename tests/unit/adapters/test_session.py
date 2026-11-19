from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AnyMessage

from planner.adapters.session import Done, Reply, Session, Stage, Token, Usage
from planner.domain.models import BaseData, Pace, SessionState


class ScriptedModel:
    def __init__(self, *script: AIMessage) -> None:
        self._script: Iterator[AIMessage] = iter(script)
        self.history_sizes: list[int] = []

    def invoke(self, messages: Sequence[AnyMessage], /) -> AIMessage:
        self.history_sizes.append(len(messages))
        return next(self._script)


def call(name: str, args: dict[str, object]) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": "c1"}])


def test_send_returns_the_assistant_text(state: SessionState, base: BaseData) -> None:
    session = Session(ScriptedModel(AIMessage("Hello")), base, "rules", state, max_rejections=3)
    assert session.send("hi") == Reply("Hello")


def test_state_is_the_start_state_until_a_tool_changes_it(
    state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(call("add_task", {"title": "Post parcel"}), AIMessage("Added."))
    session = Session(model, base, "rules", state, max_rejections=3)
    assert session.state == state
    session.send("add a parcel task")
    assert session.state.task("T6") is not None


def test_later_turns_keep_the_history(state: SessionState, base: BaseData) -> None:
    model = ScriptedModel(AIMessage("One"), AIMessage("Two"))
    session = Session(model, base, "rules", state, max_rejections=3)
    session.send("first")
    assert session.send("second") == Reply("Two")
    # rules + snapshot around 1 message, then around 3 (user, assistant, user)
    assert model.history_sizes == [3, 5]


def test_delete_asks_then_resume_decides(state: SessionState, base: BaseData) -> None:
    model = ScriptedModel(call("delete_task", {"task_id": "T4"}), AIMessage("Deleted."))
    session = Session(model, base, "rules", state, max_rejections=3)
    asked = session.send("delete T4")
    assert asked.text == ""
    assert asked.confirm is not None
    assert "T4" in asked.confirm
    assert session.resume(approved=True) == Reply("Deleted.")
    assert session.state.task("T4") is None


def test_declining_keeps_the_task(state: SessionState, base: BaseData) -> None:
    model = ScriptedModel(call("delete_task", {"task_id": "T4"}), AIMessage("Kept."))
    session = Session(model, base, "rules", state, max_rejections=3)
    session.send("delete T4")
    session.resume(approved=False)
    assert session.state == state


def test_call_log_is_empty_until_a_tool_is_called(state: SessionState, base: BaseData) -> None:
    session = Session(ScriptedModel(AIMessage("Hello")), base, "rules", state, max_rejections=3)
    assert session.call_log == []
    session.send("hi")
    assert session.call_log == []


def test_call_log_accumulates_across_turns(state: SessionState, base: BaseData) -> None:
    model = ScriptedModel(
        call("add_task", {"title": "A"}),
        AIMessage("One."),
        call("add_task", {"title": "B"}),
        AIMessage("Two."),
    )
    session = Session(model, base, "rules", state, max_rejections=3)
    session.send("first")
    session.send("second")
    assert [r.stage for r in session.call_log] == ["check", "act", "check", "act"]


def changes(state: SessionState) -> dict[str, Any]:
    return {
        "now": datetime(2026, 11, 20, 9, 0),
        "location": next(p for p in ("home", "office") if p != state.location),
        "pace": Pace(daily_limit_min=120),
    }


@pytest.mark.parametrize("field", ["now", "location", "pace"])
@pytest.mark.parametrize("started", [False, True])
def test_update_session_changes_only_that_field(
    state: SessionState, base: BaseData, field: str, *, started: bool
) -> None:
    session = Session(ScriptedModel(AIMessage("Hi")), base, "rules", state, max_rejections=3)
    if started:
        session.send("hi")
    value = changes(state)[field]
    session.update_session(**{field: value})
    expected = state.model_copy(update={field: value})
    assert session.state == expected


def test_update_session_shows_in_the_next_prompt_and_keeps_history(
    state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(call("add_task", {"title": "A"}), AIMessage("One."), AIMessage("Two."))
    session = Session(model, base, "rules", state, max_rejections=3)
    session.send("first")
    log = session.call_log
    session.update_session(pace=Pace(daily_limit_min=120))
    session.send("second")
    assert session.call_log == log
    assert session.state.task("T6") is not None
    assert session.state.pace.daily_limit_min == 120


def test_update_session_rejects_an_unknown_place(state: SessionState, base: BaseData) -> None:
    session = Session(ScriptedModel(), base, "rules", state, max_rejections=3)
    with pytest.raises(ValueError, match="unknown place"):
        session.update_session(location="narnia")


def test_update_session_waits_for_a_pending_confirmation(
    state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(call("delete_task", {"task_id": "T4"}), AIMessage("Kept."))
    session = Session(model, base, "rules", state, max_rejections=3)
    session.send("delete T4")
    with pytest.raises(RuntimeError, match="pending confirmation"):
        session.update_session(pace=Pace(daily_limit_min=120))
    session.resume(approved=False)
    session.update_session(pace=Pace(daily_limit_min=120))


def test_stream_yields_tokens_then_done(state: SessionState, base: BaseData) -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage("Hello there")]))
    session = Session(model, base, "rules", state, max_rejections=3)
    events = list(session.stream("hi"))
    assert "".join(e.text for e in events if isinstance(e, Token)) == "Hello there"
    assert events[-1] == Done(Reply("Hello there"))


def test_a_plain_model_streams_its_reply_as_one_token(state: SessionState, base: BaseData) -> None:
    session = Session(ScriptedModel(AIMessage("Hello")), base, "rules", state, max_rejections=3)
    assert list(session.stream("hi")) == [Token("Hello"), Done(Reply("Hello"))]


def test_stream_yields_each_stage_as_it_happens(state: SessionState, base: BaseData) -> None:
    model = ScriptedModel(call("delete_task", {"task_id": "T4"}), AIMessage("Deleted."))
    session = Session(model, base, "rules", state, max_rejections=3)
    first = list(session.stream("delete T4"))
    assert [e.record.stage for e in first if isinstance(e, Stage)] == ["check"]
    assert isinstance(first[-1], Done)
    assert first[-1].reply.confirm is not None
    second = list(session.resume_stream(approved=True))
    stages = [e.record for e in second if isinstance(e, Stage)]
    assert (stages[0].stage, stages[0].outcome) == ("confirm", "approved")
    assert stages[1].stage == "act"
    assert second[-1] == Done(Reply("Deleted."))


def test_done_sums_token_usage_over_the_turn(state: SessionState, base: BaseData) -> None:
    def used(message: AIMessage, tokens: int) -> AIMessage:
        usage = {"input_tokens": tokens, "output_tokens": 1, "total_tokens": tokens + 1}
        return message.model_copy(update={"usage_metadata": usage})

    model = ScriptedModel(used(call("add_task", {"title": "A"}), 10), used(AIMessage("Ok."), 20))
    session = Session(model, base, "rules", state, max_rejections=3)
    *_, done = session.stream("add A")
    assert done == Done(Reply("Ok."), Usage(input_tokens=30, output_tokens=2))


def test_done_sums_cached_tokens_and_treats_a_missing_report_as_zero(
    state: SessionState, base: BaseData
) -> None:
    def used(message: AIMessage, details: dict[str, int] | None) -> AIMessage:
        usage: dict[str, Any] = {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11}
        if details is not None:
            usage["input_token_details"] = details
        return message.model_copy(update={"usage_metadata": usage})

    model = ScriptedModel(
        used(call("add_task", {"title": "A"}), {"cache_read": 4}), used(AIMessage("Ok."), None)
    )
    session = Session(model, base, "rules", state, max_rejections=3)
    *_, done = session.stream("add A")
    assert done == Done(Reply("Ok."), Usage(input_tokens=20, output_tokens=2, cached_tokens=4))
