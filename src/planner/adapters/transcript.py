"""A record of one tab's conversation, for download: every turn, every call stage, the settings.

The custom endpoint's API key is never part of it: settings name the endpoint by URL and model.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from planner.adapters.session import Usage
from planner.app.calls import CallRecord
from planner.domain.models import Pace, SessionState, Task

if TYPE_CHECKING:
    from planner.adapters.config import Endpoint


class Turn(BaseModel):
    """One message or confirmation answer and everything it led to.

    Attributes:
        said: The user's message, or their answer to a confirmation.
        stages: Every stage of every tool call in the turn, rejected ones included.
        reply: The assistant's reply; empty while a confirmation is pending.
        confirm: The question put to the user, if the turn paused for one.
        usage: Tokens the model reported, if any.
        cost: The backend's token or cost note, if it gave one.
        seconds: How long the turn took.
        error: What the user was told when the backend failed, if it did.
    """

    said: str
    stages: list[CallRecord] = Field(default_factory=list)
    reply: str = ""
    confirm: str | None = None
    usage: Usage | None = None
    cost: str | None = None
    seconds: float = 0
    error: str | None = None


class BackendInfo(BaseModel):
    """Which model answered, without its key."""

    kind: Literal["default", "custom"]
    base_url: str | None = None
    model: str | None = None


class Settings(BaseModel):
    """What the session was set to when the transcript was taken."""

    backend: BackendInfo
    clock: Literal["real", "simulated"]
    now: datetime
    location: str
    pace: Pace


class Transcript(BaseModel):
    """Everything needed to review a conversation afterwards."""

    exported_at: datetime
    settings: Settings
    tasks: tuple[Task, ...]
    turns: list[Turn]


def transcript(
    state: SessionState,
    turns: list[Turn],
    *,
    custom: Endpoint | None,
    real_clock: bool,
    exported_at: datetime,
) -> Transcript:
    """Gather a transcript of the conversation so far.

    Args:
        state: The session as it is now.
        turns: Every turn, oldest first.
        custom: The visitor's own endpoint, or None for the default model.
        real_clock: Whether the session follows the real clock rather than a simulated one.
        exported_at: When the transcript is taken.

    Returns:
        The transcript. It holds the custom endpoint's URL and model but never its key.
    """
    backend = (
        BackendInfo(kind="default")
        if custom is None
        else BackendInfo(kind="custom", base_url=custom.base_url, model=custom.model)
    )
    settings = Settings(
        backend=backend,
        clock="real" if real_clock else "simulated",
        now=state.now,
        location=state.location,
        pace=state.pace,
    )
    return Transcript(exported_at=exported_at, settings=settings, tasks=state.tasks, turns=turns)
