"""Routes a validated tool call to the handler that carries it out."""

from __future__ import annotations

from typing import TYPE_CHECKING

from planner.app.tools import handlers
from planner.app.tools.args import (
    AddTaskArgs,
    CheckFeasibilityArgs,
    DeleteTaskArgs,
    PlanDayArgs,
    RecommendNowArgs,
    SetLocationArgs,
    ToolArgs,
    TravelTimeArgs,
    UpdateTaskArgs,
)

if TYPE_CHECKING:
    from planner.app.tools.handlers import ToolResult
    from planner.domain.models import BaseData, SessionState


def run_tool(args: ToolArgs, state: SessionState, base: BaseData) -> ToolResult:  # noqa: PLR0911 - one return per tool
    """Carry out a call that passed `validate_call`.

    Args:
        args: The typed arguments from an `Accepted` verdict.
        state: The current session state.
        base: The read-only reference data.

    Returns:
        The new state and the message for the model.
    """
    match args:
        case AddTaskArgs():
            return handlers.add_task(state, args)
        case UpdateTaskArgs():
            return handlers.update_task(state, args)
        case DeleteTaskArgs():
            return handlers.delete_task(state, args)
        case TravelTimeArgs():
            return handlers.travel_time(state, base, args)
        case CheckFeasibilityArgs():
            return handlers.check_feasibility(state, base, args)
        case PlanDayArgs():
            return handlers.plan_day(state, base, args)
        case RecommendNowArgs():
            return handlers.recommend_now(state, base)
        case SetLocationArgs():
            return handlers.set_location(state, base, args)
        case _:  # pragma: no cover - every registered tool is handled above
            msg = f"unhandled tool: {type(args).__name__}"
            raise AssertionError(msg)
