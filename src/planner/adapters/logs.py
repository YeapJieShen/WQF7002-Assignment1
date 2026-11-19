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
from collections import deque
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Literal, TextIO

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT: Final = "planner"
REDACTED: Final = "[secret]"
# Keys of the most recent custom endpoints; older ones have long since left the logs' window.
_SECRETS: Final[deque[str]] = deque(maxlen=256)
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


def session_hash(session_id: str) -> str:
    """Return a short stable hash of a session id, safe to log."""
    return hashlib.sha256(session_id.encode()).hexdigest()[:12]


def register_secret(secret: str) -> None:
    """Make every later log line replace `secret` with a marker."""
    if secret and secret not in _SECRETS:
        _SECRETS.append(secret)


def scrub(text: str) -> str:
    """Return `text` without any registered secret."""
    for secret in _SECRETS:
        text = text.replace(secret, REDACTED)
    return text


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
        return scrub(json.dumps(payload, default=str))


class TextFormatter(logging.Formatter):
    """Readable lines for a terminal: the usual prefix, the event, then key=value fields."""

    def __init__(self) -> None:
        """Use the standard time, level and logger prefix."""
        super().__init__("%(asctime)s %(levelname)-7s %(name)s %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        """Return `record` as one scrubbed text line."""
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
