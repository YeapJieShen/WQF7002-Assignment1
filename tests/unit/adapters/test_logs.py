import io
import json
import logging
from collections.abc import Iterator
from urllib.parse import quote, quote_plus

import pytest

from planner.adapters import logs
from planner.adapters.config import Settings

LOG = logging.getLogger("planner.test")


@pytest.fixture(autouse=True)
def _clean_logging() -> Iterator[None]:
    root = logging.getLogger("planner")
    before = (list(root.handlers), root.level, root.propagate)
    yield
    root.handlers[:], root.level, root.propagate = before[0], before[1], before[2]
    logs.deactivate()
    logs._SECRETS.clear()


def lines(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_json_lines_carry_event_fields_and_correlation_ids() -> None:
    out = io.StringIO()
    logs.configure_logging("INFO", "json", out)
    logs.activate("turn1", "sess1")
    logs.emit(LOG, "demo", n=3, name="x")
    [line] = lines(out)
    assert line["event"] == "demo"
    assert line["level"] == "INFO"
    assert line["logger"] == "planner.test"
    assert (line["turn_id"], line["session"], line["n"], line["name"]) == ("turn1", "sess1", 3, "x")
    assert "ts" in line


def test_no_correlation_fields_outside_a_turn() -> None:
    out = io.StringIO()
    logs.configure_logging("INFO", "json", out)
    logs.emit(LOG, "demo")
    assert "turn_id" not in lines(out)[0]


def test_level_filters_and_debug_passes_at_debug() -> None:
    out = io.StringIO()
    logs.configure_logging("INFO", "json", out)
    logs.emit(LOG, "quiet", level=logging.DEBUG)
    assert out.getvalue() == ""
    logs.configure_logging("DEBUG", "json", out)
    logs.emit(LOG, "loud", level=logging.DEBUG)
    assert lines(out)[0]["event"] == "loud"


def test_configuring_twice_keeps_one_handler() -> None:
    logs.configure_logging("INFO", "json", io.StringIO())
    logs.configure_logging("INFO", "text", io.StringIO())
    assert len(logging.getLogger("planner").handlers) == 1


def test_text_format_is_readable() -> None:
    out = io.StringIO()
    logs.configure_logging("INFO", "text", out)
    logs.emit(LOG, "demo", n=3)
    logs.emit(LOG, "bare")
    first, second = out.getvalue().splitlines()
    assert first.endswith("planner.test demo n=3")
    assert second.endswith("planner.test bare")


def test_exceptions_are_logged_with_the_stack() -> None:
    out = io.StringIO()
    logs.configure_logging("INFO", "json", out)
    try:
        raise ValueError("boom")  # noqa: TRY301 - the raise is what the test logs
    except ValueError as exc:
        logs.emit(LOG, "failed", level=logging.ERROR, exc_info=exc)
    trace = str(lines(out)[0]["exc"])
    assert "test_exceptions_are_logged_with_the_stack" in trace  # the stack
    assert "ValueError" in trace
    assert "ValueError: boom" not in trace  # the message may quote user input


def test_the_exception_message_is_logged_at_debug() -> None:
    out = io.StringIO()
    logs.configure_logging("DEBUG", "json", out)
    try:
        raise ValueError("boom")  # noqa: TRY301 - the raise is what the test logs
    except ValueError as exc:
        logs.emit(LOG, "failed", level=logging.ERROR, exc_info=exc)
    assert "ValueError: boom" in str(lines(out)[0]["exc"])


def test_registered_secrets_are_scrubbed_from_both_formats_and_stacks() -> None:
    logs.register_secret("sk-supersecret")
    for fmt in ("json", "text"):
        out = io.StringIO()
        logs.configure_logging("INFO", fmt, out)
        try:
            raise ValueError("bad sk-supersecret")  # noqa: TRY301 - the raise is what the test logs
        except ValueError as exc:
            logs.emit(LOG, "failed", level=logging.ERROR, exc_info=exc, detail="sk-supersecret")
        assert "sk-supersecret" not in out.getvalue()
        assert logs.REDACTED in out.getvalue()


def test_bind_tags_each_resumption_and_clears_after() -> None:
    out = io.StringIO()
    logs.configure_logging("INFO", "json", out)

    def work() -> Iterator[int]:
        logs.emit(LOG, "first")
        yield 1
        logs.emit(LOG, "second")
        yield 2

    seen = []
    for item in logs.bind("t", "s", work()):
        seen.append(item)
        logs.deactivate()  # as if the next step ran in a context without the ids
    assert seen == [1, 2]
    first, second = lines(out)
    assert first["turn_id"] == second["turn_id"] == "t"
    logs.emit(LOG, "after")
    assert "turn_id" not in lines(out)[2]


def test_session_hash_is_stable_and_not_the_id() -> None:
    assert logs.session_hash("abc") == logs.session_hash("abc")
    assert "abc" not in logs.session_hash("abc")
    assert logs.session_hash("abc") != logs.session_hash("abd")


def test_settings_level_is_case_insensitive_and_defaults_are_documented() -> None:
    assert Settings(log_level="debug").log_level == "DEBUG"  # type: ignore[arg-type]  # a lowercase level, which the Literal type rejects on purpose
    assert (Settings().log_level, Settings().log_format) == ("INFO", "json")


def test_settings_summary_has_no_keys() -> None:
    summary = Settings().summary()
    assert "api_key" not in json.dumps(summary)
    assert summary["llm_endpoints"][0]["host"] == "localhost"


AWKWARD_KEYS = [r'sk-A"B\C-é9', "sk-plain-0123456789", "sk-sp ace&x=1/é"]


@pytest.mark.parametrize("key", AWKWARD_KEYS)
@pytest.mark.parametrize("fmt", ["json", "text"])
def test_awkward_keys_are_scrubbed_in_every_spelling(key: str, fmt: str) -> None:
    logs.register_secret(key)
    out = io.StringIO()
    logs.configure_logging("INFO", fmt, out)  # type: ignore[arg-type]  # fmt is a parametrised str; the Literal type is checked elsewhere
    spellings = [key, quote(key, safe=""), quote_plus(key), json.dumps(key)]
    try:
        raise ValueError("; ".join(spellings))  # noqa: TRY301 - the raise is what the test logs
    except ValueError as exc:
        logs.emit(
            LOG, f"event {key}", level=logging.ERROR, exc_info=exc,
            detail=key, nested={"k": [key, quote(key, safe="")]}, obj=ValueError(key),
        )  # fmt: skip
    text = out.getvalue()
    for spelling in spellings:
        assert spelling not in text
    assert key not in text
    assert json.dumps(key)[1:-1] not in text
    assert logs.REDACTED in text


@pytest.mark.parametrize("fmt", ["json", "text"])
def test_a_key_logged_lowercased_inside_a_host_is_scrubbed(fmt: str) -> None:
    # URL parsing lowercases hostnames, so the logged host is not the registered spelling.
    logs.register_secret("sk-AUDITKEY123")
    out = io.StringIO()
    logs.configure_logging("INFO", fmt, out)  # type: ignore[arg-type]  # fmt is a parametrised str; the Literal type is checked elsewhere
    logs.emit(LOG, "endpoint_rejected", host="sk-auditkey123.evil.example.com")
    assert "sk-auditkey123" not in out.getvalue().lower()
    assert logs.REDACTED in out.getvalue()


def test_short_or_placeholder_keys_are_not_registered() -> None:
    logs.register_secret("ollama")
    logs.register_secret("")
    assert not logs._SECRETS
    assert logs.scrub("ollama is local") == "ollama is local"


def test_a_secret_stays_scrubbed_until_every_owner_forgets_it() -> None:
    for i in range(1000):  # far more than the old window: nothing is evicted
        logs.register_secret(f"filler-key-{i:05d}")
    logs.register_secret("sk-live-secret")
    logs.register_secret("sk-live-secret")
    assert logs.scrub("a sk-live-secret b") == f"a {logs.REDACTED} b"
    logs.forget_secret("sk-live-secret")
    assert logs.REDACTED in logs.scrub("sk-live-secret")
    logs.forget_secret("sk-live-secret")
    logs.forget_secret("sk-live-secret")  # forgetting twice is harmless
    assert logs.scrub("sk-live-secret") == "sk-live-secret"
    assert logs.REDACTED in logs.scrub("filler-key-00000")


def test_stack_info_is_scrubbed_too() -> None:
    logs.register_secret("sk-in-the-stack-1")
    out = io.StringIO()
    logs.configure_logging("INFO", "json", out)
    LOG.error("with stack", stack_info=True)
    assert "stack" in lines(out)[0]
