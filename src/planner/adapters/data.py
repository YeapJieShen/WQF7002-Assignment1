"""Loads the packaged base data and sample personas. Read-only: nothing here writes."""

from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING, Final

from planner.domain.models import BaseData, SessionState

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

PERSONA: Final = "persona_wednesday"  # the sample persona the app and the evals start from
_RESOURCES: Final[Traversable] = files("planner") / "resources"


def load_base() -> BaseData:
    """Load places, travel table and planning config.

    Returns:
        The immutable base data.

    Raises:
        pydantic.ValidationError: If the file is malformed or the travel table names a
            place that is not listed.
    """
    return BaseData.model_validate_json((_RESOURCES / "base.json").read_text("utf-8"))


def load_rules() -> str:
    """Load the fixed rules of the system prompt."""
    return (_RESOURCES / "rules.md").read_text("utf-8")


def list_personas() -> list[str]:
    """List the names of the sample personas, in alphabetical order."""
    return sorted(
        p.name.removesuffix(".json")
        for p in (_RESOURCES / "personas").iterdir()
        if p.name.endswith(".json")
    )


def load_persona(name: str) -> SessionState:
    """Load a sample persona's starting session state.

    Args:
        name: File name without extension, e.g. "persona_wednesday".

    Returns:
        The starting session state.
    """
    return SessionState.model_validate_json(
        (_RESOURCES / "personas" / f"{name}.json").read_text("utf-8")
    )


def persona_label(state: SessionState) -> str:
    """Name a persona by its simulated start, such as "Wednesday 28 Oct, 15:40"."""
    return f"{state.now:%A} {state.now.day} {state.now:%b, %H:%M}"


def persona_choices() -> list[tuple[str, str]]:
    """Every sample persona as a (label, name) pair for a dropdown, ordered by start time."""
    loaded = sorted(((load_persona(n), n) for n in list_personas()), key=lambda p: p[0].now)
    return [(persona_label(state), name) for state, name in loaded]
