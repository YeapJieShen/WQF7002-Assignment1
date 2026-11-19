"""Structured logging for the app: one `planner` logger tree, JSON lines, one-time setup.

Every module logs through `logging.getLogger("planner.<area>")` with `emit`, which attaches the
event's fields under `extra`. `configure_logging` is called once by an entry point (the app or the
eval runner), never at import time and never from `domain` or `app`.

Privacy rules this module backs up (each has a test):
    * An API key is never logged. Keys of custom endpoints are registered with `register_secret`
      and every formatted line is scrubbed of them, stack traces included.
    * User message text and tool arguments are logged only at DEBUG.
    * A visitor is identified by a hash of the session id, never the raw id.

A turn's events share a turn id and session hash through context variables, which `bind` sets
again each time the turn's generator is resumed: Gradio may resume it on a different thread.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Literal, TextIO
from urllib.parse import quote, quote_plus

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT: Final = "planner"
REDACTED: Final = "[secret]"
# Shorter values are placeholders such as the "ollama" key; scrubbing them would mangle ordinary
# words in the logs, while a real API key is far longer.
MIN_SECRET_LEN: Final = 8
# Secret -> number of owners (tabs, operator settings) still using it. A secret is scrubbed for as
# long as it has an owner and is never evicted for being old; `forget_secret` releases one owner.
_SECRETS: Final[dict[str, int]] = {}
_SECRETS_LOCK: Final = threading.Lock()
_TURN: ContextVar[str | None] = ContextVar("planner_turn", default=None)
_SESSION: ContextVar[str | None] = ContextVar("planner_session", default=None)
_HANDLER_MARK: Final = "_planner_handler"

type LogFormat = Literal["json", "text"]


def emit(
    logger: logging.Logger,
    event: str,
    /,
    *,
    level: int = logging.INFO,
    exc_info: BaseException | bool | None = None,
    **fields: object,
) -> None:
    """Log `event` with `fields`, which the formatters print as separate keys."""
    logger.log(level, event, extra={"fields": fields}, exc_info=exc_info, stacklevel=2)


def elapsed_ms(started: float) -> float:
    """Return the milliseconds since `started`, a `time.perf_counter()` reading."""
    return round((time.perf_counter() - started) * 1000, 2)


def session_hash(session_id: str) -> str:
    """Return a short stable hash of a session id, safe to log."""
    return hashlib.sha256(session_id.encode()).hexdigest()[:12]


def register_secret(secret: str) -> None:
    """Make every later log line replace `secret` with a marker, until each owner forgets it.

    Values shorter than `MIN_SECRET_LEN` are ignored: they are placeholders, not keys.
    """
    if len(secret) >= MIN_SECRET_LEN:
        with _SECRETS_LOCK:
            _SECRETS[secret] = _SECRETS.get(secret, 0) + 1


def forget_secret(secret: str) -> None:
    """Release one owner of `secret`; it stops being scrubbed once no owner is left."""
    with _SECRETS_LOCK:
        if (owners := _SECRETS.get(secret, 0)) > 1:
            _SECRETS[secret] = owners - 1
        else:
            _SECRETS.pop(secret, None)


def _spellings(secret: str) -> set[str]:
    """Return the ways `secret` can appear in a line: as is, JSON-escaped, URL-encoded."""
    return {secret, json.dumps(secret)[1:-1], quote(secret, safe=""), quote_plus(secret)}


def scrub(text: str) -> str:
    """Return `text` without any registered secret, in any of its spellings."""
    with _SECRETS_LOCK:
        secrets = list(_SECRETS)
    # Longest first, so a secret that contains another is not left half replaced.
    for spelling in sorted({v for s in secrets for v in _spellings(s)}, key=len, reverse=True):
        text = text.replace(spelling, REDACTED)
    return text


def _clean(value: object) -> object:
    """Return `value` with every string in it scrubbed; objects become their scrubbed `str`."""
    match value:
        case str():
            return scrub(value)
        case bool() | int() | float() | None:
            return value
        case dict():
            return {scrub(str(k)): _clean(v) for k, v in value.items()}
        case list() | tuple() | set() | frozenset():
            return [_clean(v) for v in value]
        case _:
            return scrub(str(value))


def activate(turn_id: str, session: str) -> None:
    """Tag the events logged from here, in this context, with a turn and session."""
    _TURN.set(turn_id)
    _SESSION.set(session)


def deactivate() -> None:
    """Stop tagging events with a turn."""
    _TURN.set(None)
    _SESSION.set(None)


def bind[T](turn_id: str, session: str, events: Iterator[T]) -> Iterator[T]:
    """Yield `events`, tagging everything logged while each item is produced."""
    try:
        while True:
            activate(turn_id, session)
            try:
                item = next(events)
            except StopIteration:
                return
            yield item
    finally:
        deactivate()


class Correlation(logging.Filter):
    """Stamps each record with the current turn id and session hash."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add `turn_id` and `session` to `record`; never drops it."""
        record.turn_id = _TURN.get()
        record.session = _SESSION.get()
        return True


def _fields(record: logging.LogRecord) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if turn := getattr(record, "turn_id", None):
        out["turn_id"] = turn
    if session := getattr(record, "session", None):
        out["session"] = session
    out.update(getattr(record, "fields", {}))
    return out


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, event, correlation ids, fields, error."""

    def format(self, record: logging.LogRecord) -> str:
        """Return `record` as one scrubbed JSON line."""
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            **_fields(record),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        # Scrub the values first: once serialised, a key with a quote, backslash or non-ASCII
        # character no longer matches. The final scrub catches whatever the first one cannot see.
        return scrub(json.dumps(_clean(payload)))


class TextFormatter(logging.Formatter):
    """Readable lines for a terminal: the usual prefix, the event, then key=value fields."""

    def __init__(self) -> None:
        """Use the standard time, level and logger prefix."""
        super().__init__("%(asctime)s %(levelname)-7s %(name)s %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        """Return `record` as one scrubbed text line, exception and stack included."""
        line = super().format(record)
        extras = " ".join(f"{k}={v}" for k, v in _fields(record).items())
        return scrub(f"{line} {extras}" if extras else line)


def configure_logging(
    level: str = "INFO", log_format: LogFormat = "json", stream: TextIO | None = None
) -> None:
    """Send the `planner` loggers to `stream` (stderr by default); safe to call again."""
    root = logging.getLogger(ROOT)
    for old in [h for h in root.handlers if getattr(h, _HANDLER_MARK, False)]:
        root.removeHandler(old)
    handler = logging.StreamHandler(stream)
    setattr(handler, _HANDLER_MARK, True)
    handler.addFilter(Correlation())
    handler.setFormatter(JsonFormatter() if log_format == "json" else TextFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
