# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
import os
from collections.abc import Callable
from typing import Final

import pytest

from planner.adapters.config import Settings
from planner.adapters.data import load_base, load_persona
from planner.app.validation import Accepted, validate_call
from planner.domain.errors import ErrorCode
from planner.domain.models import BaseData, SessionState

Codes = Callable[[object, object], list[ErrorCode]]
SAID: Final = "Add it, about 45 minutes, by Friday"
"""What the user said this turn in tests that need a stated duration or deadline."""


@pytest.fixture(scope="session")
def base() -> BaseData:
    return load_base()


@pytest.fixture(scope="session")
def state() -> SessionState:
    return load_persona("persona_wednesday")


@pytest.fixture
def codes(state: SessionState, base: BaseData) -> Codes:
    """Validate a call against persona_wednesday, the user having said `SAID`; return its codes."""

    def run(name: object, args: object) -> list[ErrorCode]:
        verdict = validate_call(name, args, state, base, said=(SAID,))
        return [] if isinstance(verdict, Accepted) else [i.code for i in verdict.issues]

    return run


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's `.env` and `PLANNER_*` variables out of every test."""
    for name in [n for n in os.environ if n.startswith("PLANNER_")]:
        monkeypatch.delenv(name)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
