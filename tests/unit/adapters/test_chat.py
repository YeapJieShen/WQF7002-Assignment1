import threading
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from planner.adapters.chat import Tab, Tick, _heartbeat, new_tab, stream_turn, use_endpoint
from planner.adapters.config import Endpoint
from planner.adapters.session import Event, Token
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
            tab,
            backend,
            history,
            lambda: tab.session.stream(text),
            said=text,
            signed_in=signed_in,
            rotate_s=60,
        )
    )


def titles(chat: list[Any]) -> list[str]:
    return [(m.get("metadata") or {}).get("title", "") for m in chat]


def test_tokens_grow_one_bubble(state: SessionState, base: BaseData) -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage("Hello there")]))
    backend, tab = make(model, state, base)
    frames = run(tab, backend, "hi")
    assert len(frames) > 2
    assert frames[-1][-1]["content"].startswith("Hello there\n\n")
    assert len([m for m in frames[-1] if m["role"] == "assistant"]) == 1


def test_the_cost_line_follows_the_reply_and_is_recorded(
    state: SessionState, base: BaseData
) -> None:
    usage = {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}
    backend, tab = make(ScriptedModel(AIMessage("Hello", usage_metadata=usage)), state, base)
    reply = run(tab, backend, "hi")[-1][-1]["content"]
    assert reply.startswith("Hello\n\n*12 tokens*\n\n")
    assert reply.endswith("5 in / 7 out tokens</span>")  # this reply's own seconds and tokens
    assert tab.turns[0].cost == "12 tokens"
    assert tab.turns[0].usage is not None
    assert tab.turns[0].seconds >= 0


def test_each_stage_is_its_own_row_and_the_reply_comes_last(
    state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(call("add_task", {"title": "Post parcel"}), AIMessage("Added."))
    backend, tab = make(model, state, base, price=None)
    final = run(tab, backend, "add")[-1]
    assert [t.split(" <")[0] for t in titles(final) if t] == ["add_task"]  # one row for the call
    assert 'badge ok">done' in titles(final)[1]
    assert final[-1]["content"].startswith("Added.\n\n")
    assert [r.stage for r in tab.turns[0].stages] == ["check", "act"]


def test_a_rejected_call_is_shown_and_recorded_with_its_reason(
    state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(call("set_location", {"place_id": "mars"}), AIMessage("Sorry."))
    backend, tab = make(model, state, base, price=None)
    final = run(tab, backend, "I am on mars")[-1]
    assert any("set_location" in t and 'badge bad">rejected' in t for t in titles(final))
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
            lambda: tab.session.resume_stream(approved=True),
            said="(confirmed)",
            signed_in=False,
            rotate_s=60,
        )
    )[-1]
    assert final[-1]["content"].startswith("Deleted.")
    assert [t.said for t in tab.turns] == ["delete T4", "(confirmed)"]
    assert [t.split(" <")[0] for t in titles(final) if t] == ["delete_task"]
    assert 'badge ok">done' in titles(final)[-2]


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


def test_releasing_an_endpoint_stops_scrubbing_its_key_only_when_unused() -> None:
    from pydantic import SecretStr  # noqa: PLC0415 - only this test needs it

    from planner.adapters import logs  # noqa: PLC0415 - only this test needs the logs module
    from planner.adapters.chat import release_endpoint  # noqa: PLC0415 - only this test needs it

    mine = Endpoint(
        name="c", base_url="https://x.test/v1", model="m", api_key=SecretStr("sk-release-me-123")
    )
    logs._SECRETS.clear()
    logs.register_secret("sk-release-me-123")
    logs.register_secret("sk-release-me-123")
    release_endpoint(None)
    release_endpoint(mine)
    assert logs.scrub("sk-release-me-123") == logs.REDACTED
    release_endpoint(mine)
    assert logs.scrub("sk-release-me-123") == "sk-release-me-123"


def answer(tab: Tab, backend: FakeBackend, history: list[Any]) -> list[Any]:
    tab.pending = None
    return list(
        stream_turn(
            tab,
            backend,
            history,
            lambda: tab.session.resume_stream(approved=True),
            said="(confirmed)",
            signed_in=False,
            rotate_s=60,
        )
    )[-1]


def test_an_answer_updates_the_waiting_row_in_place(state: SessionState, base: BaseData) -> None:
    model = ScriptedModel(call("delete_task", {"task_id": "T4"}), AIMessage("Deleted."))
    backend, tab = make(model, state, base, price=None)
    asked = run(tab, backend, "delete T4")[-1]
    assert "waiting for you" in titles(asked)[-1]
    final = answer(tab, backend, asked)
    rows = [t for t in titles(final) if "delete_task" in t]
    assert len(rows) == 1
    assert 'badge ok">done' in rows[0]


def test_an_answer_works_when_the_browser_sends_no_metadata_on_plain_messages(
    state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(call("delete_task", {"task_id": "T4"}), AIMessage("Deleted."))
    backend, tab = make(model, state, base, price=None)
    asked = run(tab, backend, "delete T4")[-1]
    sent = [{**m, "metadata": m.get("metadata")} for m in asked]  # plain messages carry None
    assert any(m["metadata"] is None for m in sent)
    final = answer(tab, backend, sent)
    assert next(t for t in titles(final) if "delete_task" in t).count("badge ok") == 1


def test_the_pending_line_shows_first_then_names_the_running_tool(
    state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(call("add_task", {"title": "Post parcel"}), AIMessage("Added."))
    backend, tab = make(model, state, base, price=None)
    frames = run(tab, backend, "add")
    assert "Thinking…" in frames[0][-1]["content"]
    assert any("Running add_task…" in f[-1]["content"] for f in frames[:-1])
    assert "spinner" not in str(frames[-1])


def test_a_slow_model_rotates_the_pending_text() -> None:
    gate = threading.Event()

    def slow() -> Iterator[Event]:
        gate.wait(2)
        yield Token("hi")

    seen: list[Tick] = []
    for item in _heartbeat(slow(), 0.01):
        if isinstance(item, Tick) and len(seen) < 2:
            seen.append(item)
        if len(seen) == 2:
            gate.set()
    assert len(seen) == 2


def test_an_error_in_the_model_thread_reaches_the_consumer() -> None:
    def broken() -> Iterator[Event]:
        yield Token("hi")
        raise Boom

    with pytest.raises(Boom):
        list(_heartbeat(broken(), 1))
