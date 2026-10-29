"""Loads the packaged base data and sample personas. Read-only: nothing here writes."""

from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING, Final

from planner.domain.models import BaseData, SessionState

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

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
