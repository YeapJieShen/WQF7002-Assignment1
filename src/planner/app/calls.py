# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
"""The audit trail of tool calls: one entry for each stage a call passes through."""

from __future__ import annotations

from typing import Literal

from planner.domain.models import Frozen


class CallRecord(Frozen):
    """One stage of one tool call, as shown in the call log.

    Attributes:
        stage: `check` is validation, `confirm` is the user's answer, `act` is the run.
        tool: The tool name as the model gave it.
        args: The arguments as JSON text.
        outcome: What happened at this stage, for example `accepted` or `rejected: ...`.
    """

    stage: Literal["check", "confirm", "act"]
    tool: str
    args: str
    outcome: str
