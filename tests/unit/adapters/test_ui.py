import inspect
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import pytest
from langchain_core.messages import AIMessage

from planner.adapters.chat import Tab
from planner.adapters.config import Settings
from planner.adapters.graph import GAVE_UP
from planner.adapters.ui import (
    COPY_JS,
    FILE_TOO_LARGE,
    HELP_HTML,
    MARK_PASTE_JS,
    MAX_MIN,
    MESSAGE_TOO_LONG,
    PASTE_MARK,
    SESSION_LOST,
    USAGE_SPAN,
    WATCH_PASTE_JS,
    Controller,
    build_app,
    pace_from,
    pace_values,
    real_now,
    size_text,
)
from planner.adapters.views import SAMPLE_DAYS, USAGE_HINT
from planner.domain.models import Pace, Weekday
from tests.unit.adapters.fakes import Boom, FailingModel, FakeBackend, ScriptedModel, call

NOW = datetime(2026, 11, 19, 9, 30)
KEY = "sk-super-secret-key"


def settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


def controller(model: Any = None, **overrides: Any) -> Controller:
    backend = FakeBackend(model or ScriptedModel(), price=None)
    return Controller(backend, settings(**overrides), clock=lambda: NOW, rng=random.Random(1))


def boot(ctl: Controller) -> Tab:
    tab = ctl.boot()[0]
    assert isinstance(tab, Tab)
    return tab


def chat(ctl: Controller, tab: Tab | None, text: str, profile: Any = None) -> list[Any]:
    return list(ctl.chat(text, [], tab, profile))


def reply(chat: list[dict[str, Any]]) -> str:
    """The last message's text without its usage footer."""
    content: str = chat[-1]["content"]
    return content.split('\n\n<span class="turn-usage"', maxsplit=1)[0]


def test_a_new_tab_starts_empty_on_the_real_clock() -> None:
    ctl = controller()
    tab, chatbot, box, tasks, chip, card, question, *rest = ctl.boot()
    assert tab.real_clock is True
    assert tab.session.state.now == NOW
    assert tab.session.state.tasks == ()
    assert tasks == []
    assert chip == "📍 Home"
    assert len(chatbot.examples) == 3
    assert box["interactive"] is True
    assert card["visible"] is False
    assert question == ""
    assert len(rest) == 13
    assert len(rest[-1].data[1].x) == SAMPLE_DAYS + 1


def test_try_a_demo_loads_the_persona_on_a_simulated_clock() -> None:
    ctl = controller()
    tab = ctl.demo(boot(ctl))[0]
    assert tab.real_clock is False
    assert tab.session.state.tasks
    assert tab.session.state.now == datetime(2026, 10, 28, 15, 40)


def test_try_a_demo_loads_the_chosen_persona() -> None:
    ctl = controller()
    tab = ctl.demo(boot(ctl), "persona_saturday")[0]
    assert tab.session.state.now == datetime(2026, 10, 31, 9, 30)


def test_a_restart_keeps_the_visitors_own_endpoint() -> None:
    ctl = controller()
    tab = boot(ctl)
    ctl.connect(tab, "custom", "https://x.test/v1", "m", KEY)
    assert ctl.demo(tab)[0].custom == tab.custom


def test_a_restart_the_backend_refuses_is_reported() -> None:
    ctl = controller()
    tab = boot(ctl)
    ctl.connect(tab, "custom", "https://x.test/v1", "m", KEY)
    assert tab.custom is not None
    tab.custom = tab.custom.model_copy(update={"base_url": "http://plain"})
    with pytest.raises(gr.Error, match="https"):
        ctl.demo(tab)


def test_chat_streams_the_reply_and_refresh_shows_the_new_tasks() -> None:
    ctl = controller(ScriptedModel(call("add_task", {"title": "Post parcel"}), AIMessage("Added.")))
    tab = boot(ctl)
    frames = chat(ctl, tab, "add a parcel task")
    assert frames[0][0] == [{"role": "user", "content": "add a parcel task"}]
    assert frames[0][1]["interactive"] is False
    assert reply(frames[-1][0]) == "Added."
    box, tasks, chip, card, question = ctl.refresh(tab)
    assert any(row[1] == "Post parcel" for row in tasks)
    assert box["interactive"] is True
    assert card["visible"] is False
    assert (chip, question) == ("📍 Home", "")
    assert re.search(
        r'<span class="turn-usage" title="[^"]+">\d+\.\d s</span>$', frames[-1][0][-1]["content"]
    )


def test_blank_and_whitespace_messages_do_nothing() -> None:
    ctl = controller()
    tab = boot(ctl)
    for text in ("", "   ", PASTE_MARK + " \n"):
        assert len(chat(ctl, tab, text)) == 1
    assert tab.turns == []


def test_the_real_clock_follows_the_clock_on_each_message() -> None:
    times = [NOW, datetime(2026, 11, 19, 10, 0)]
    ctl = Controller(
        FakeBackend(ScriptedModel(AIMessage("ok"))),
        settings(),
        clock=lambda: times.pop(0),
        rng=random.Random(1),
    )
    tab = boot(ctl)
    chat(ctl, tab, "hi")
    assert tab.session.state.now == datetime(2026, 11, 19, 10, 0)


def test_a_delete_shows_the_card_then_the_answer_settles_it() -> None:
    ctl = controller(ScriptedModel(call("delete_task", {"task_id": "T4"}), AIMessage("Deleted.")))
    tab = ctl.demo(None)[0]
    history = chat(ctl, tab, "delete T4")[-1][0]
    box, _, _, card, question = ctl.refresh(tab)
    assert card["visible"] is True
    assert "T4" in question
    assert box["interactive"] is False
    assert reply(list(ctl.approve(tab, history))[-1]) == "Deleted."
    box, tasks, _, card, _ = ctl.refresh(tab)
    assert all(row[0] != "T4" for row in tasks)
    assert card["visible"] is False
    assert box["interactive"] is True


def test_declining_keeps_the_task() -> None:
    ctl = controller(ScriptedModel(call("delete_task", {"task_id": "T4"}), AIMessage("Kept.")))
    tab = ctl.demo(None)[0]
    history = chat(ctl, tab, "delete T4")[-1][0]
    list(ctl.decline(tab, history))
    assert tab.session.state.task("T4") is not None
    assert tab.turns[-1].said == "(declined)"


def test_a_backend_failure_is_explained_and_knows_whether_the_visitor_is_signed_in() -> None:
    ctl = controller(FailingModel(Boom()))
    tab = boot(ctl)
    assert chat(ctl, tab, "hi")[-1][0][-1]["content"] == "Out of quota."
    chat(ctl, tab, "again", profile=object())
    assert ctl.backend.explained == [(False, False), (True, False)]  # type: ignore[attr-defined]  # the fake backend records calls; the protocol omits explained


def test_a_pasted_message_is_labelled_and_fenced() -> None:
    ctl = controller(ScriptedModel(AIMessage("It asks for lunch.")))
    tab = boot(ctl)
    frames = chat(ctl, tab, PASTE_MARK + "Lunch at 1?")
    assert frames[0][0][0]["content"] == "(forwarded)\nLunch at 1?"
    assert tab.turns[0].said == "Lunch at 1?"


def test_a_lost_tab_asks_for_a_reload() -> None:
    ctl = controller()
    for attempt in (
        lambda: ctl.refresh(None),
        lambda: ctl.export_state(None),
        lambda: ctl.export_transcript(None),
        lambda: ctl.connect(None, "default", "", "", ""),
        lambda: ctl.pace_changed(None),
        lambda: list(ctl.approve(None, [])),
    ):
        with pytest.raises(gr.Error, match=SESSION_LOST):
            attempt()


def test_connecting_a_custom_endpoint_holds_the_key_on_the_server_only() -> None:
    ctl = controller()
    tab = boot(ctl)
    status, cleared = ctl.connect(tab, "custom", " https://x.test/v1 ", "gpt-x", KEY)
    assert tab.custom is not None
    assert tab.custom.api_key.get_secret_value() == KEY
    assert KEY not in status
    assert cleared == ""
    assert ctl.connect(tab, "default", "", "", "")[0] == "Using the default model."
    assert tab.custom is None


def test_a_refused_endpoint_is_reported_and_changes_nothing() -> None:
    ctl = controller()
    tab = boot(ctl)
    with pytest.raises(gr.Error, match="https"):
        ctl.connect(tab, "custom", "http://x.test", "m", KEY)
    with pytest.raises(gr.Error):
        ctl.connect(tab, "custom", "https://x.test", "", KEY)
    assert tab.custom is None


def test_the_pace_controls_update_the_session_and_the_chart() -> None:
    ctl = controller()
    tab = ctl.demo(None)[0]
    values = list(pace_values(tab.session.state.pace))
    values[0] = 120
    chart = ctl.pace_changed(tab, *values)
    assert tab.session.state.pace.daily_limit_min == 120
    assert chart.data[0].y[0] <= 120


def test_pace_changes_wait_for_a_pending_confirmation() -> None:
    ctl = controller(ScriptedModel(call("delete_task", {"task_id": "T4"})))
    tab = ctl.demo(None)[0]
    chat(ctl, tab, "delete T4")
    values = list(pace_values(tab.session.state.pace))
    values[0] = 60
    ctl.pace_changed(tab, *values)
    assert tab.session.state.pace.daily_limit_min != 60


def test_every_pace_field_round_trips_through_the_controls() -> None:
    pace = Pace(
        daily_limit_min=300,
        max_consecutive_min=90,
        long_break_min=20,
        weekday_limit_min={Weekday.SAT: 60},
        distribution=0.2,
        hardest_first=True,
    )
    daily, consecutive, long_break, distribution, hardest, *weekdays = pace_values(pace)
    again = pace_from(
        daily,
        consecutive,
        long_break,
        distribution,
        hardest_first=hardest,
        weekdays=weekdays,
        current=Pace(),
    )
    assert again == pace


def test_an_emptied_pace_field_keeps_its_value_and_zero_turns_breaks_off() -> None:
    current = Pace(daily_limit_min=300, long_break_min=20, distribution=0.2)
    pace = pace_from(None, 0, None, None, hardest_first=False, weekdays=[None] * 7, current=current)
    assert (pace.daily_limit_min, pace.long_break_min, pace.distribution) == (300, 20, 0.2)
    assert pace.max_consecutive_min is None


def test_an_exported_session_imports_back() -> None:
    ctl = controller()
    tab = ctl.demo(None)[0]
    path = ctl.export_state(tab)["value"]
    again = ctl.import_state(tab, path)[0]
    assert again.session.state == tab.session.state
    assert again.real_clock is False


@pytest.mark.parametrize("content", ["not json", '{"now": "2026-10-29T09:00", "location": "mars"}'])
def test_a_bad_import_is_reported(content: str, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(content, "utf-8")
    with pytest.raises(gr.Error, match="Could not import"):
        controller().import_state(None, str(bad))
    with pytest.raises(gr.Error, match="Could not import"):
        controller().import_state(None, None)


def test_the_transcript_has_everything_but_never_the_key() -> None:
    ctl = controller()
    tab = boot(ctl)
    ctl.connect(tab, "custom", "https://x.test/v1", "gpt-x", KEY)
    tab.switch.model = ScriptedModel(
        call("set_location", {"place_id": "mars"}), AIMessage("Sorry, unknown.")
    )
    chat(ctl, tab, "I am on mars")
    path = ctl.export_transcript(tab)["value"]
    text = Path(path).read_text("utf-8")
    assert KEY not in text
    assert KEY not in Path(ctl.export_state(tab)["value"]).read_text("utf-8")
    data = json.loads(text)
    assert data["settings"]["backend"] == {
        "kind": "custom",
        "base_url": "https://x.test/v1",
        "model": "gpt-x",
    }
    assert data["settings"]["clock"] == "real"
    assert [s["stage"] for s in data["turns"][0]["stages"]] == ["check"]
    assert data["turns"][0]["stages"][0]["outcome"].startswith("rejected")
    assert data["turns"][0]["reply"] == "Sorry, unknown."


def test_picking_an_example_fills_the_box() -> None:
    ctl = controller()
    assert ctl.pick_example(gr.SelectData(None, {"index": 0, "value": {"text": "Plan"}})) == "Plan"
    assert ctl.pick_example(gr.SelectData(None, {"index": 0, "value": "hi"})) == "hi"


def test_real_now_is_naive_local_time() -> None:
    assert real_now().tzinfo is None


def test_the_page_scripts_share_the_mark_and_skip_the_plain_text_paste() -> None:
    assert json.dumps(PASTE_MARK) in MARK_PASTE_JS
    assert "!e.shiftKey" in WATCH_PASTE_JS


def test_exactly_one_endpoint_is_public_and_the_key_is_not_among_its_inputs() -> None:
    app = build_app(FakeBackend(ScriptedModel()), settings())
    public = [f for f in app.fns.values() if f.api_visibility == "public"]
    assert [f.api_name for f in public] == ["chat"]
    key = next(b for b in app.blocks.values() if getattr(b, "label", None) == "API key")
    assert key._id not in [c._id for c in public[0].inputs]
    assert {type(c).__name__ for c in public[0].inputs} == {"Textbox", "Chatbot", "State"}


def test_the_confirmation_buttons_stream_into_the_chat() -> None:
    # Gradio streams a handler only if it is itself a generator function; given a plain function
    # that returns a generator, it posts the generator object to the chat, which then fails.
    ctl = controller()
    app = build_app(FakeBackend(ScriptedModel()), settings(), controller=ctl)
    answers = [
        f for f in app.fns.values() if f.fn is not None and f.fn.__name__ in {"approve", "decline"}
    ]
    assert len(answers) == 2
    assert all(inspect.isgeneratorfunction(f.fn) for f in answers)


def test_a_caller_without_a_page_gets_a_tab_of_its_own() -> None:
    ctl = controller(ScriptedModel(AIMessage("Hello.")))
    frames = chat(ctl, None, "hi")
    assert isinstance(frames[0][2], Tab)
    assert reply(frames[-1][0]) == "Hello."


# --- limits --------------------------------------------------------------------------------


def test_the_agent_gives_up_after_the_configured_rejections() -> None:
    model = ScriptedModel(*[call("nope", {}) for _ in range(5)])
    ctl = controller(model, max_rejections=1)
    frames = chat(ctl, boot(ctl), "do it")
    assert reply(frames[-1][0]) == GAVE_UP
    assert len(list(model._script)) == 4  # one call used, not three


def test_a_message_over_the_limit_is_refused_with_both_numbers() -> None:
    ctl = controller(message_max_chars=10)
    tab = boot(ctl)
    frames = chat(ctl, tab, "x" * 5210)
    assert len(frames) == 1
    note = frames[0][0][-1]["content"]
    assert note == MESSAGE_TOO_LONG.format(actual=5210, limit=10)
    assert "5,210 characters; the limit is 10. Shorten it or split it." in note
    assert tab.turns == []


def test_a_message_at_the_limit_goes_through() -> None:
    ctl = controller(ScriptedModel(AIMessage("ok")), message_max_chars=10)
    assert reply(chat(ctl, boot(ctl), "x" * 10)[-1][0]) == "ok"


def test_the_message_box_and_the_queue_use_the_settings() -> None:
    app = build_app(
        FakeBackend(ScriptedModel()),
        settings(message_max_chars=123, queue_max_size=7, concurrency_limit=3, state_ttl_s=99),
    )
    box = next(b for b in app.blocks.values() if getattr(b, "elem_id", None) == "message-box")
    assert getattr(box, "max_length", None) == 123
    assert app._queue.max_size == 7
    chat_fns = [f for f in app.fns.values() if f.name == "chat"]
    assert chat_fns
    assert {f.concurrency_limit for f in chat_fns} == {3}
    state = next(b for b in app.blocks.values() if type(b).__name__ == "State")
    assert getattr(state, "time_to_live", None) == 99


def test_a_file_over_the_limit_is_refused_with_both_sizes(tmp_path: Path) -> None:
    big = tmp_path / "big.json"
    big.write_bytes(b" " * 3_200_000)
    with pytest.raises(gr.Error) as caught:
        controller(import_max_bytes=1_000_000).import_state(None, str(big))
    assert caught.value.message == FILE_TOO_LARGE.format(actual="3.2 MB", limit="1.0 MB")
    assert "File is 3.2 MB; the limit is 1.0 MB." in caught.value.message


def test_sizes_read_as_a_person_would_say_them() -> None:
    assert (size_text(512), size_text(3_200_000)) == ("0.5 KB", "3.2 MB")


def test_a_custom_endpoint_gets_the_custom_timeout() -> None:
    ctl = controller(custom_timeout_s=12)
    tab = boot(ctl)
    ctl.connect(tab, "custom", "https://x.test/v1", "m", KEY)
    assert tab.custom is not None
    assert tab.custom.timeout_s == 12


def _caption(block: Any) -> str:
    return str(getattr(block, "value", None) or getattr(block, "label", ""))


def test_the_toolbar_has_the_file_buttons_and_the_demo_lives_in_the_sidebar() -> None:
    app = build_app(FakeBackend(ScriptedModel()), settings())
    blocks = list(app.blocks.values())
    [sidebar] = [b for b in blocks if b.get_block_name() == "sidebar"]
    assert "Try a demo" in {_caption(b) for b in blocks if b.parent is sidebar}
    toolbar = {
        _caption(b) for b in blocks if getattr(b.parent, "elem_classes", None) == ["toolbar"]
    }
    assert toolbar == {"Save session", "Load session", "Download chat log"}


def test_choosing_the_default_model_applies_at_once_and_hides_the_endpoint_fields() -> None:
    ctl = controller()
    tab = boot(ctl)
    ctl.connect(tab, "custom", "https://x.test/v1", "m", KEY)
    fields, status = ctl.choose(tab, "default")
    assert fields["visible"] is False
    assert status == "Using the default model."
    assert tab.custom is None
    fields, status = ctl.choose(tab, "custom")
    assert fields["visible"] is True
    assert tab.custom is None  # nothing changes until the visitor presses the button


def test_the_model_choice_is_a_dropdown_and_the_charge_notice_hides_behind_a_help_mark() -> None:
    app = build_app(FakeBackend(ScriptedModel()), settings())
    blocks = list(app.blocks.values())
    offered = [
        [value for _, value in getattr(b, "choices", [])]
        for b in blocks
        if b.get_block_name() == "dropdown"
    ]
    assert ["default", "custom"] in offered
    assert ["persona_wednesday", "persona_saturday"] in offered  # the demo's personas
    assert not any(
        "charge you" in str(getattr(b, "value", "")) and b.get_block_name() == "markdown"
        for b in blocks
    )
    assert "charge you" in HELP_HTML


def test_the_pace_controls_accept_exactly_what_the_domain_accepts() -> None:
    assert MAX_MIN == 24 * 60
    assert Pace(daily_limit_min=MAX_MIN).daily_limit_min == MAX_MIN
    with pytest.raises(ValueError, match="less than or equal"):
        Pace(daily_limit_min=MAX_MIN + 1)


def test_a_restart_releases_the_old_tabs_claim_on_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released: list[Any] = []
    monkeypatch.setattr("planner.adapters.ui.release_endpoint", released.append)
    ctl = controller()
    tab = boot(ctl)
    ctl.connect(tab, "custom", "https://x.test/v1", "m", KEY)
    ctl.demo(tab)
    assert released == [tab.custom]


def test_reset_to_defaults_restores_every_pace_control_and_the_session() -> None:
    ctl = controller()
    tab = boot(ctl)
    values = (60, 25, 10, 0.2, True, *([30] * 7))
    ctl.pace_changed(tab, *values)
    *controls, chart = ctl.reset_pace(tab)
    assert tab.session.state.pace == Pace()
    assert tuple(controls) == pace_values(Pace())
    assert len(chart.data[1].x) == SAMPLE_DAYS + 1


def test_the_weekday_limits_are_always_visible_with_one_reset_button() -> None:
    app = build_app(FakeBackend(ScriptedModel()), settings())
    blocks = list(app.blocks.values())
    numbers = [getattr(b, "label", "") for b in blocks if b.get_block_name() == "number"]
    assert numbers[-7:] == [d.value.title() for d in Weekday]
    assert not any(b.get_block_name() == "accordion" and "weekday" in _caption(b) for b in blocks)
    assert [getattr(b, "value", None) for b in blocks].count("Reset to defaults") == 1


def test_the_copy_wrapper_strips_only_the_usage_span() -> None:
    reply = f'Hello.\n\n<span class="turn-usage" title="{USAGE_HINT}">1 In / 2 Out Tokens</span>'
    assert re.sub(USAGE_SPAN, "", reply) == "Hello."
    other = 'Keep <span class="turn-usage-x">this</span> and <span>this</span>'
    assert re.sub(USAGE_SPAN, "", other) == other
    assert json.dumps(USAGE_SPAN) in COPY_JS
