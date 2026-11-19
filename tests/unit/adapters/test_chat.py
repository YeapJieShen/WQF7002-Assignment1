from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from planner.adapters.chat import Tab, new_tab, stream_turn, use_endpoint
from planner.adapters.config import Endpoint
from planner.domain.models import BaseData, SessionState
from tests.unit.adapters.fakes import Boom, FailingModel, FakeBackend, ScriptedModel, call


def make(
    model: Any, state: SessionState, base: BaseData, *, price: str | None = "12 tokens"
) -> tuple[FakeBackend, Tab]:
    backend = FakeBackend(model, price=price)
    return backend, new_tab(backend, base, "rules", state, real_clock=False, max_rejections=3)


def run(tab: Tab, backend: FakeBackend, text: str, *, signed_in: bool = False) -> list[Any]:
    history = [{"role": "user", "content": text}]
    return list(
        stream_turn(
            tab, backend, history, lambda: tab.session.stream(text), said=text, signed_in=signed_in
        )
    )


def titles(chat: list[Any]) -> list[str]:
    return [m.get("metadata", {}).get("title", "") for m in chat]


def test_tokens_grow_one_bubble(state: SessionState, base: BaseData) -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage("Hello there")]))
    backend, tab = make(model, state, base)
    frames = run(tab, backend, "hi")
    assert len(frames) > 2
    assert frames[-1][-1]["content"] == "Hello there"
    assert len([m for m in frames[-1] if m["role"] == "assistant"]) == 1


def test_the_cost_line_follows_the_reply_and_is_recorded(
    state: SessionState, base: BaseData
) -> None:
    usage = {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}
    backend, tab = make(ScriptedModel(AIMessage("Hello", usage_metadata=usage)), state, base)
    assert run(tab, backend, "hi")[-1][-1]["content"] == "Hello\n\n*12 tokens*"
    assert tab.turns[0].cost == "12 tokens"
    assert tab.turns[0].usage is not None
    assert tab.turns[0].seconds >= 0


def test_each_stage_is_its_own_row_and_the_reply_comes_last(
    state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(call("add_task", {"title": "Post parcel"}), AIMessage("Added."))
    backend, tab = make(model, state, base, price=None)
    final = run(tab, backend, "add")[-1]
    assert any("add_task · check" in t for t in titles(final))
    assert any("add_task · act" in t for t in titles(final))
    assert final[-1]["content"] == "Added."
    assert [r.stage for r in tab.turns[0].stages] == ["check", "act"]


def test_a_rejected_call_is_shown_and_recorded_with_its_reason(
    state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(call("set_location", {"place_id": "mars"}), AIMessage("Sorry."))
    backend, tab = make(model, state, base, price=None)
    final = run(tab, backend, "I am on mars")[-1]
    assert any("❌" in t and "set_location" in t for t in titles(final))
    assert tab.turns[0].stages[0].outcome.startswith("rejected")


def test_a_confirmation_pauses_then_the_answer_finishes_it(
    state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(call("delete_task", {"task_id": "T4"}), AIMessage("Deleted."))
    backend, tab = make(model, state, base, price=None)
    run(tab, backend, "delete T4")
    assert tab.pending is not None
    assert "T4" in tab.pending
    tab.pending = None
    final = list(
        stream_turn(
            tab,
            backend,
            [],
            lambda: tab.session.resume_stream(True),  # noqa: FBT003
            said="(confirmed)",
            signed_in=False,
        )
    )[-1]
    assert final[-1]["content"] == "Deleted."
    assert [t.said for t in tab.turns] == ["delete T4", "(confirmed)"]
    assert any("confirm: approved" in t for t in titles(final))


def test_an_explained_failure_becomes_a_message_and_is_recorded(
    state: SessionState, base: BaseData
) -> None:
    backend, tab = make(FailingModel(Boom()), state, base)
    final = run(tab, backend, "hi", signed_in=True)[-1]
    assert final[-1]["content"] == "Out of quota."
    assert tab.turns[0].error == "Out of quota."
    assert backend.explained == [(True, False)]


def test_an_unexplained_failure_is_raised_but_still_recorded(
    state: SessionState, base: BaseData
) -> None:
    backend, tab = make(FailingModel(RuntimeError("bug")), state, base)
    with pytest.raises(RuntimeError, match="bug"):
        run(tab, backend, "hi")
    assert len(tab.turns) == 1


def test_switching_endpoint_changes_the_model_mid_conversation(
    state: SessionState, base: BaseData
) -> None:
    backend, tab = make(ScriptedModel(AIMessage("default")), state, base)
    mine = Endpoint(name="custom", base_url="https://x.test/v1", model="m")
    use_endpoint(tab, backend, mine)
    assert tab.custom == mine
    with pytest.raises(ValueError, match="https"):
        use_endpoint(tab, backend, Endpoint(name="c", base_url="http://x", model="m"))
    assert tab.custom == mine
    use_endpoint(tab, backend, None)
    assert tab.custom is None
