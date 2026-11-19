# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
"""Rules about when the agent must stop and ask the user before acting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from planner.app.tools.args import DeleteTaskArgs
from planner.app.validation import DELETE_ALL

if TYPE_CHECKING:
    from planner.app.tools.args import ToolArgs
    from planner.domain.models import SessionState


def confirmation_question(args: ToolArgs, state: SessionState) -> str | None:
    """Return the yes/no question to put to the user, or None if the call needs no confirmation.

    Only deletion asks first, because it cannot be undone. The call must already have passed
    `validate_call`.

    Args:
        args: The typed arguments from an `Accepted` verdict.
        state: The current session state.

    Returns:
        The question, or None when the tool can run straight away.
    """
    if not isinstance(args, DeleteTaskArgs):
        return None
    if args.task_id == DELETE_ALL:
        return f"Delete all {len(state.tasks)} tasks? This cannot be undone."
    task = state.task(args.task_id)
    label = f"{args.task_id} {task.title}" if task else args.task_id
    return f"Delete {label}? This cannot be undone."
