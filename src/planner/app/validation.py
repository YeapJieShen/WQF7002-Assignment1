"""Validation of model tool calls before anything runs.

Two stages, both deterministic:
1. Shape: allowlist, then the tool's argument model (types, ranges, no unexpected arguments).
2. Meaning: checks that need the current state (places, task IDs, horizon, dependency cycles).

A rejected call carries issues phrased for the model to correct; it is never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import TYPE_CHECKING, Final

from pydantic import ValidationError

from planner.app.tools.args import (
    TOOLS_BY_NAME,
    AddTaskArgs,
    CheckFeasibilityArgs,
    DeleteTaskArgs,
    PlanDayArgs,
    RecommendNowArgs,
    ToolArgs,
    TravelTimeArgs,
    UpdateTaskArgs,
)
from planner.domain.errors import ErrorCode, Issue
from planner.domain.models import ANYWHERE, BaseData, SessionState

if TYPE_CHECKING:
    from datetime import date, datetime

    from pydantic_core import ErrorDetails

DELETE_ALL: Final = "all"
"""Value of `delete_task.task_id` that targets every task."""

_CODE_BY_PYDANTIC_TYPE: Final[dict[str, ErrorCode]] = {
    "missing": ErrorCode.MISSING_ARG,
    "extra_forbidden": ErrorCode.UNEXPECTED_ARG,
    "greater_than_equal": ErrorCode.OUT_OF_RANGE,
    "less_than_equal": ErrorCode.OUT_OF_RANGE,
    "enum": ErrorCode.OUT_OF_RANGE,
    "string_too_short": ErrorCode.BAD_VALUE,
    "datetime_parsing": ErrorCode.BAD_DATETIME,
    "datetime_from_date_parsing": ErrorCode.BAD_DATETIME,
    "datetime_type": ErrorCode.BAD_DATETIME,
    "timezone_naive": ErrorCode.BAD_DATETIME,
    "date_parsing": ErrorCode.BAD_DATETIME,
    "date_from_datetime_parsing": ErrorCode.BAD_DATETIME,
    "date_from_datetime_inexact": ErrorCode.BAD_DATETIME,
    "date_type": ErrorCode.BAD_DATETIME,
}


@dataclass(frozen=True, slots=True)
class Accepted:
    """A call that passed validation.

    Attributes:
        args: The parsed, typed arguments.
    """

    args: ToolArgs


@dataclass(frozen=True, slots=True)
class Rejected:
    """A call that failed validation.

    Attributes:
        issues: What was wrong, each phrased so the model can correct it.
    """

    issues: tuple[Issue, ...]

    def message(self) -> str:
        """Join the issues into one string to return to the model."""
        return "; ".join(str(i) for i in self.issues)


Verdict = Accepted | Rejected


def reject_malformed(name: str | None, error: str) -> Rejected:
    """Wrap a LangChain invalid_tool_call so it is rejected visibly instead of dropped."""
    return Rejected((Issue(ErrorCode.MALFORMED_CALL, name or "name", error or "unparseable call"),))


def validate_call(name: object, raw_args: object, state: SessionState, base: BaseData) -> Verdict:
    """Validate one tool call: allowlist, shape, then meaning against the current state.

    Args:
        name: Tool name as given by the model.
        raw_args: Arguments as given by the model.
        state: The current session state.
        base: The read-only reference data.

    Returns:
        `Accepted` with typed arguments, or `Rejected` with the issues found.
    """
    tool = TOOLS_BY_NAME.get(name) if isinstance(name, str) else None
    if tool is None:
        allowed = ", ".join(sorted(TOOLS_BY_NAME))
        return Rejected(
            (Issue(ErrorCode.UNKNOWN_TOOL, "name", f"not on the allowlist ({allowed})"),)
        )
    if not isinstance(raw_args, dict):
        return Rejected((Issue(ErrorCode.BAD_TYPE, "args", "must be an object"),))
    try:
        args = tool.model_validate(raw_args)
    except ValidationError as exc:
        return Rejected(tuple(_issue_from_pydantic(e) for e in exc.errors()))
    issues = _MeaningChecker(state, base).check(args)
    return Rejected(tuple(issues)) if issues else Accepted(args)


def creates_cycle(state: SessionState, task_id: str, prerequisites: tuple[str, ...]) -> bool:
    """Tell whether giving a task these prerequisites makes it depend on itself.

    Args:
        state: The current session state.
        task_id: The task being edited.
        prerequisites: The prerequisites it would have after the edit.

    Returns:
        True if the prerequisite graph, with the edit applied, contains a cycle.
    """
    graph = {t.id: set(t.prerequisites) for t in state.tasks}
    graph[task_id] = set(prerequisites)
    try:
        TopologicalSorter(graph).prepare()
    except CycleError:
        return True
    return False


def _issue_from_pydantic(error: ErrorDetails) -> Issue:
    kind = error["type"]
    try:
        code = ErrorCode(kind)  # custom errors are raised with an ErrorCode value as type
    except ValueError:
        code = _CODE_BY_PYDANTIC_TYPE.get(kind, ErrorCode.BAD_TYPE)
    location = ".".join(str(part) for part in error["loc"])
    return Issue(code, location or "args", error["msg"])


@dataclass(frozen=True, slots=True)
class _MeaningChecker:
    """Stage 2: checks that need the current state and reference data."""

    state: SessionState
    base: BaseData

    def check(self, args: ToolArgs) -> list[Issue]:  # noqa: PLR0911 - one return per tool
        """Return every semantic problem with already well-shaped arguments."""
        match args:
            case AddTaskArgs():
                return [
                    *self._in_horizon(args.deadline, "deadline"),
                    *self._in_horizon(args.earliest_start, "earliest_start"),
                    *self._known_place(args.place, "place", allow_anywhere=True),
                    *self._known_tasks(args.prerequisites, "prerequisites"),
                ]
            case UpdateTaskArgs():
                return self._check_update(args)
            case DeleteTaskArgs():
                if args.task_id == DELETE_ALL:
                    return []
                return self._known_tasks((args.task_id,), "task_id")
            case TravelTimeArgs():
                return [
                    *self._known_place(args.from_place, "from_place", allow_anywhere=False),
                    *self._known_place(args.to_place, "to_place", allow_anywhere=False),
                ]
            case CheckFeasibilityArgs():
                return [
                    *self._known_tasks((args.task_id,), "task_id"),
                    *self._in_horizon(args.start, "start"),
                    *self._known_place(args.from_place, "from_place", allow_anywhere=False),
                ]
            case PlanDayArgs():
                return self._day_in_horizon(args.date)
            case RecommendNowArgs():
                return []
            case _:  # pragma: no cover - every registered tool is handled above
                msg = f"unhandled tool: {type(args).__name__}"
                raise AssertionError(msg)

    def _check_update(self, args: UpdateTaskArgs) -> list[Issue]:
        patch = args.fields
        issues = [
            *self._known_tasks((args.task_id,), "task_id"),
            *self._in_horizon(patch.deadline, "fields.deadline"),
            *self._in_horizon(patch.earliest_start, "fields.earliest_start"),
            *self._known_place(patch.place, "fields.place", allow_anywhere=True),
        ]
        if patch.prerequisites is not None:
            issues += self._known_tasks(patch.prerequisites, "fields.prerequisites")
            if not issues and creates_cycle(self.state, args.task_id, patch.prerequisites):
                issues.append(
                    Issue(
                        ErrorCode.CYCLE, "fields.prerequisites", "would create a dependency cycle"
                    )
                )
        return issues

    def _in_horizon(self, value: datetime | None, field: str) -> list[Issue]:
        now = self.state.now
        end = self.base.config.horizon_end(now)
        if value is None or now <= value <= end:
            return []
        window = f"{now:%Y-%m-%dT%H:%M} to {end:%Y-%m-%dT%H:%M}"
        return [Issue(ErrorCode.OUT_OF_HORIZON, field, f"must be within {window}")]

    def _day_in_horizon(self, day: date) -> list[Issue]:
        now = self.state.now
        if now.date() <= day <= self.base.config.horizon_end(now).date():
            return []
        return [Issue(ErrorCode.OUT_OF_HORIZON, "date", "outside the planning horizon")]

    def _known_place(self, value: str | None, field: str, *, allow_anywhere: bool) -> list[Issue]:
        if value is None or value in self.base.place_ids or (allow_anywhere and value == ANYWHERE):
            return []
        known = ", ".join(sorted(self.base.place_ids))
        extra = ", anywhere" if allow_anywhere else ""
        return [
            Issue(ErrorCode.UNKNOWN_PLACE, field, f"'{value}' is not known (known: {known}{extra})")
        ]

    def _known_tasks(self, ids: tuple[str, ...], field: str) -> list[Issue]:
        return [
            Issue(ErrorCode.UNKNOWN_TASK, field, f"no task with id '{task_id}'")
            for task_id in ids
            if task_id not in self.state.task_ids
        ]
