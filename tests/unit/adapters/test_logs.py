import io
import json
import logging
from collections.abc import Iterator

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
        raise ValueError("boom")  # noqa: TRY301
    except ValueError as exc:
        logs.emit(LOG, "failed", level=logging.ERROR, exc_info=exc)
    assert "ValueError: boom" in str(lines(out)[0]["exc"])


def test_registered_secrets_are_scrubbed_from_both_formats_and_stacks() -> None:
    logs.register_secret("sk-supersecret")
    for fmt in ("json", "text"):
        out = io.StringIO()
        logs.configure_logging("INFO", fmt, out)
        try:
            raise ValueError("bad sk-supersecret")  # noqa: TRY301
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
    assert Settings(log_level="debug").log_level == "DEBUG"  # type: ignore[arg-type]
    assert (Settings().log_level, Settings().log_format) == ("INFO", "json")


def test_settings_summary_has_no_keys() -> None:
    summary = Settings().summary()
    assert "api_key" not in json.dumps(summary)
    assert summary["llm_endpoints"][0]["host"] == "localhost"
