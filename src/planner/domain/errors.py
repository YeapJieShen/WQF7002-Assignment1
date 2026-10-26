"""Typed error vocabulary shared by validation, policy checks and tools."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ErrorCode(StrEnum):
    """Machine-readable reason a tool call was rejected."""

    UNKNOWN_TOOL = "unknown_tool"
    MALFORMED_CALL = "malformed_call"
    UNEXPECTED_ARG = "unexpected_arg"
    MISSING_ARG = "missing_arg"
    BAD_TYPE = "bad_type"
    BAD_VALUE = "bad_value"
    OUT_OF_RANGE = "out_of_range"
    BAD_DATETIME = "bad_datetime"
    OUT_OF_HORIZON = "out_of_horizon"
    UNKNOWN_PLACE = "unknown_place"
    UNKNOWN_TASK = "unknown_task"
    CYCLE = "cycle"


@dataclass(frozen=True, slots=True)
class Issue:
    """One problem with a tool call, phrased so the model can correct it."""

    code: ErrorCode
    field: str
    message: str

    def __str__(self) -> str:
        """Render as `field: message [code]` for the model to read."""
        return f"{self.field}: {self.message} [{self.code}]"
