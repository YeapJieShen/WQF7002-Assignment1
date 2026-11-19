"""Logged events of whole turns, endpoints and limits, and the privacy rules around them."""

import io
import json
import logging
import random
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import httpx2
import openai
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

from planner.adapters import logs
from planner.adapters.backends import AppBackend, ProviderError, _Redacting
from planner.adapters.config import Endpoint, Settings
from planner.adapters.endpoint_guard import BlockedEndpointError
from planner.adapters.ui import Controller
from planner.adapters.zerogpu import TemplateChat
from tests.unit.adapters.conftest import answer
from tests.unit.adapters.fakes import Boom, FailingModel, FakeBackend, ScriptedModel, call
from tests.unit.adapters.test_log_events import everything, fields
from tests.unit.adapters.test_zerogpu import CLOSE, OPEN, FakeEngine

NOW = datetime(2026, 11, 19, 9, 30)
KEY = "sk-super-secret-key-0123"
USER_TEXT = "my private plan about Jalan Secret"
REQUEST = httpx2.Request("POST", "https://api.example.com/v1/chat/completions")


@pytest.fixture(autouse=True)
def _capture(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    caplog.set_level(logging.DEBUG, logger="planner")
    root = logging.getLogger("planner")
    before = (list(root.handlers), root.level, root.propagate)
    yield
    root.handlers[:], root.level, root.propagate = before
    logs.deactivate()
    logs._SECRETS.clear()


@pytest.fixture
def json_log() -> io.StringIO:
    """Route the planner logs through the real JSON formatter into a buffer."""
    out = io.StringIO()
    logs.configure_logging("DEBUG", "json", out)
    return out


def records(out: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in out.getvalue().splitlines()]


def controller(model: Any = None, backend: Any = None, **overrides: Any) -> Controller:
    served = backend or FakeBackend(model or ScriptedModel(), price=None)
    return Controller(
        served, Settings(_env_file=None, **overrides), clock=lambda: NOW, rng=random.Random(1)
    )


def talk(ctl: Controller, tab: Any, text: str) -> list[Any]:
    return list(ctl.chat(text, [], tab, None))


def failing(error: Exception) -> RunnableLambda[object, AIMessage]:
    def raise_it(_: object) -> AIMessage:
        raise error

    return RunnableLambda(raise_it)


# --- turns, sessions, backends -------------------------------------------------------------


def test_every_event_of_a_turn_shares_a_turn_id_and_a_hashed_session(json_log: io.StringIO) -> None:
    usage = {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}
    model = ScriptedModel(
        call("add_task", {"title": "Post parcel"}), AIMessage("Added.", usage_metadata=usage)
    )
    ctl = controller(model)
    tab = ctl.boot()[0]
    talk(ctl, tab, "add a task")
    turn = [r for r in records(json_log) if "turn_id" in r]
    assert {r["event"] for r in turn} >= {
        "turn_start",
        "turn_input",
        "graph_step",
        "backend_call",
        "tool_check",
        "tool_act",
        "turn_end",
    }
    assert len({r["turn_id"] for r in turn}) == 1
    assert {r["session"] for r in turn} == {tab.session.log_id}
    thread_id = tab.session._config["configurable"]["thread_id"]
    assert thread_id not in json_log.getvalue()
    assert logs.session_hash(thread_id) == tab.session.log_id
    end = next(r for r in turn if r["event"] == "turn_end")
    assert (end["outcome"], end["tokens_in"], end["tokens_out"]) == ("reply", 5, 7)
    assert end["n_tool_calls"] == 1
    assert end["latency_ms"] >= 0


def test_a_new_session_and_its_end_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    ctl = controller()
    tab = ctl.boot()[0]
    [start] = fields(caplog, "session_start")
    assert start["session"] == tab.session.log_id
    ctl.expired(tab)
    ctl.expired(None)
    assert fields(caplog, "session_end") == [{"session": tab.session.log_id, "turns": 0}]


def test_a_confirmation_is_asked_then_answered(caplog: pytest.LogCaptureFixture) -> None:
    ctl = controller(ScriptedModel(call("delete_task", {"task_id": "T4"}), AIMessage("Deleted.")))
    tab = ctl.demo(None)[0]
    talk(ctl, tab, "delete T4")
    assert fields(caplog, "confirm_asked") == [{"tool": "delete_task"}]
    assert [f["outcome"] for f in fields(caplog, "turn_end")] == ["confirm_pending"]
    list(ctl.approve(tab, [], None))
    assert fields(caplog, "confirm_answered") == [{"tool": "delete_task", "approved": True}]
    assert [f["outcome"] for f in fields(caplog, "turn_end")] == ["confirm_pending", "reply"]


def test_a_backend_failure_is_logged_with_class_and_status(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctl = controller(FailingModel(Boom("quota")))
    talk(ctl, ctl.boot()[0], "hi")
    [failed] = fields(caplog, "backend_error")
    assert failed["backend"] == "default"
    assert failed["error_class"] == "Boom"
    assert failed["status_code"] is None
    assert [f["outcome"] for f in fields(caplog, "turn_end")] == ["backend_error"]


def test_an_unexpected_failure_is_logged_with_its_stack_and_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctl = controller(FailingModel(ValueError("bug")))
    with pytest.raises(ValueError, match="bug"):
        talk(ctl, ctl.boot()[0], "hi")
    [failed] = [r for r in caplog.records if r.getMessage() == "turn_error"]
    assert failed.exc_info is not None
    assert failed.fields["error_class"] == "ValueError"  # type: ignore[attr-defined]
    assert [f["outcome"] for f in fields(caplog, "turn_end")] == ["error"]


def test_the_turn_context_is_cleared_when_a_turn_ends(json_log: io.StringIO) -> None:
    ctl = controller(ScriptedModel(AIMessage("ok")))
    talk(ctl, ctl.boot()[0], "hi")
    logs.emit(logging.getLogger("planner.test"), "after")
    assert "turn_id" not in records(json_log)[-1]


def test_a_custom_backend_call_names_the_host_and_model(caplog: pytest.LogCaptureFixture) -> None:
    ctl = controller(ScriptedModel(AIMessage("ok")))
    tab = ctl.boot()[0]
    ctl.connect(tab, "custom", "https://api.example.com/v1/secret?k=1", "gpt-x", KEY)
    tab.switch.model = ScriptedModel(AIMessage("ok"))
    talk(ctl, tab, "hi")
    [made] = fields(caplog, "backend_call")
    assert (made["backend"], made["host"], made["model"]) == ("custom", "api.example.com", "gpt-x")


def test_the_gpu_generation_reports_queue_wait_and_duration(
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine = FakeEngine(["Hello ", "there"])
    TemplateChat(engine=engine, call_open=OPEN, call_close=CLOSE).invoke("hi")
    [gen] = fields(caplog, "generate")
    assert gen["first_piece_ms"] is not None
    assert gen["duration_ms"] >= gen["first_piece_ms"]
    assert (gen["tokens_in"], gen["tokens_out"]) == (3, 2)


# --- endpoints and limits ------------------------------------------------------------------


def test_an_endpoint_change_logs_scheme_and_host_only(caplog: pytest.LogCaptureFixture) -> None:
    ctl = controller()
    tab = ctl.boot()[0]
    ctl.connect(tab, "custom", "https://api.example.com/v1/private/path?key=abc", "gpt-x", KEY)
    ctl.connect(tab, "default", "", "", "")
    custom, default = fields(caplog, "endpoint_change")
    assert custom == {
        "backend": "custom",
        "model": "gpt-x",
        "scheme": "https",
        "host": "api.example.com",
    }
    assert default == {"backend": "default"}
    assert "private" not in everything(caplog)
    assert "abc" not in everything(caplog)


@pytest.mark.parametrize(
    ("url", "dns", "reason"),
    [
        ("http://api.example.com/v1", "93.184.216.34", "not_https"),
        ("https://api.example.com:8443/v1", "93.184.216.34", "not_443"),
        ("https://user:pw@api.example.com/v1", "93.184.216.34", "userinfo"),
        ("https:///x", "93.184.216.34", "no_host"),
        ("https://api.example.com/v1", "10.0.0.5", "private_address"),
    ],
)
def test_a_refused_endpoint_logs_a_reason_code_and_no_secrets(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    dns: str,
    reason: str,
) -> None:
    monkeypatch.setattr("socket.getaddrinfo", answer(dns))
    backend = AppBackend(ScriptedModel(), zerogpu=False, allow_private=False, error_max_chars=300)
    with pytest.raises(BlockedEndpointError):
        backend.model(Endpoint(name="c", base_url=url, model="m", api_key=SecretStr(KEY)))
    [refused] = fields(caplog, "endpoint_rejected")
    assert refused["reason"] == reason
    assert "pw" not in everything(caplog)


def test_a_request_the_guard_refuses_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    blocked = BlockedEndpointError("inside", "private_address")
    with pytest.raises(ProviderError):
        _Redacting(failing(blocked), KEY, 30, 300).invoke([])
    assert fields(caplog, "endpoint_rejected") == [{"reason": "private_address", "at": "request"}]


def test_each_limit_hit_names_the_limit_with_actual_and_allowed(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    ctl = controller(message_max_chars=10, import_max_bytes=100, upload_max_bytes=200)
    talk(ctl, ctl.boot()[0], "x" * 50)
    big = tmp_path / "big.json"
    big.write_bytes(b" " * 150)
    with pytest.raises(gr.Error):
        ctl.import_state(None, str(big))
    hits = {f["limit"]: f for f in fields(caplog, "limit_hit")}
    assert (hits["message_max_chars"]["actual"], hits["message_max_chars"]["allowed"]) == (50, 10)
    assert (hits["import_max_bytes"]["actual"], hits["import_max_bytes"]["allowed"]) == (150, 100)


def test_provider_limits_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    slow = openai.APITimeoutError(request=REQUEST)
    with pytest.raises(ProviderError, match="answer"):
        _Redacting(failing(slow), KEY, 12, 300).invoke([])
    long = openai.APIConnectionError(message="x" * 500, request=REQUEST)
    with pytest.raises(ProviderError, match="xxx"):
        _Redacting(failing(long), KEY, 12, 50).invoke([])
    hits = {f["limit"]: f for f in fields(caplog, "limit_hit")}
    assert hits["custom_timeout_s"]["allowed"] == 12
    assert hits["provider_error_max_chars"]["allowed"] == 50
    assert hits["provider_error_max_chars"]["actual"] > 50


def test_import_export_and_transcript_log_sizes_only(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    ctl = controller()
    tab = ctl.demo(None)[0]
    exported = Path(ctl.export_state(tab)["value"])
    ctl.import_state(tab, str(exported))
    ctl.export_transcript(tab)
    bad = tmp_path / "bad.json"
    bad.write_text("not json", "utf-8")
    with pytest.raises(gr.Error):
        ctl.import_state(tab, str(bad))
    [export] = fields(caplog, "session_export")
    [imported] = fields(caplog, "session_import")
    [download] = fields(caplog, "transcript_download")
    assert export["size_bytes"] == imported["size_bytes"] == exported.stat().st_size
    assert download["size_bytes"] > 0
    assert len(fields(caplog, "import_failed")) == 1


# --- privacy -------------------------------------------------------------------------------


def custom_turn(model: Any) -> tuple[Controller, Any]:
    backend = AppBackend(ScriptedModel(), zerogpu=False, allow_private=False, error_max_chars=300)
    ctl = controller(backend=backend)
    tab = ctl.boot()[0]
    ctl.connect(tab, "custom", "https://api.example.com/v1", "gpt-x", KEY)
    tab.switch.model = _Redacting(model, KEY, 30, 300)
    return ctl, tab


@pytest.mark.usefixtures("public_dns")
def test_the_api_key_is_in_no_record_during_a_custom_endpoint_turn(
    caplog: pytest.LogCaptureFixture, json_log: io.StringIO
) -> None:
    ctl, tab = custom_turn(failing_free("fine"))
    talk(ctl, tab, USER_TEXT)
    assert caplog.records
    assert KEY not in everything(caplog)
    assert KEY not in json_log.getvalue()


def failing_free(text: str) -> RunnableLambda[object, AIMessage]:
    return RunnableLambda(lambda _: AIMessage(text))


@pytest.mark.usefixtures("public_dns")
def test_the_api_key_is_in_no_record_when_the_provider_errors(
    caplog: pytest.LogCaptureFixture, json_log: io.StringIO
) -> None:
    quoted = openai.AuthenticationError(
        f"Incorrect API key provided: {KEY}",
        response=httpx2.Response(401, request=REQUEST),
        body=None,
    )
    ctl, tab = custom_turn(failing(quoted))
    frames = talk(ctl, tab, USER_TEXT)
    assert "Incorrect API key" in frames[-1][0][-1]["content"]
    assert KEY not in everything(caplog)
    assert KEY not in json_log.getvalue()
    [failed] = fields(caplog, "backend_error")
    assert (failed["error_class"], failed["status_code"]) == ("ProviderError", 401)


@pytest.mark.usefixtures("public_dns")
def test_a_key_quoted_in_an_unexpected_error_is_scrubbed_from_the_stack(
    json_log: io.StringIO,
) -> None:
    ctl, tab = custom_turn(failing(RuntimeError(f"upstream said {KEY}")))
    with pytest.raises(RuntimeError):
        talk(ctl, tab, "hi")
    assert KEY not in json_log.getvalue()
    assert logs.REDACTED in json_log.getvalue()


def test_user_text_and_tool_arguments_are_logged_only_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def run() -> None:
        model = ScriptedModel(
            call("add_task", {"title": "Jalan Secret errand"}), AIMessage("Added your plan.")
        )
        ctl = controller(model)
        talk(ctl, ctl.boot()[0], USER_TEXT)

    caplog.set_level(logging.INFO, logger="planner")
    run()
    assert caplog.records
    for private in ("Jalan Secret", "private plan", "Added your plan"):
        assert private not in everything(caplog)
    caplog.clear()
    caplog.set_level(logging.DEBUG, logger="planner")
    run()
    assert USER_TEXT in [f["text"] for f in fields(caplog, "turn_input")]
    assert "Jalan Secret errand" in fields(caplog, "tool_args")[0]["args"]
    assert fields(caplog, "turn_reply")[0]["text"] == "Added your plan."


def test_the_start_up_summary_has_no_keys(caplog: pytest.LogCaptureFixture) -> None:
    secret = Endpoint(name="n", base_url="https://h.example/v1", model="m", api_key=SecretStr(KEY))
    summary = Settings(_env_file=None, llm_endpoints=(secret,)).summary()
    logs.emit(logging.getLogger("planner.app"), "app_start", **summary)
    assert KEY not in everything(caplog)
    assert fields(caplog, "app_start")[0]["llm_endpoints"][0]["host"] == "h.example"
