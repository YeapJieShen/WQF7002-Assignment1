# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
"""Validation of model tool calls before anything runs.

Two stages, both deterministic:
1. Shape: allowlist, then the tool's argument model (types, ranges, no unexpected arguments).
2. Meaning: checks that need the current state (places, task IDs, horizon, dependency cycles).

A rejected call carries issues phrased for the model to correct; it is never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import TYPE_CHECKING, Final, Protocol, cast

from pydantic import BaseModel, ValidationError

from planner.app.tools.args import (
    TOOLS_BY_NAME,
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
from planner.domain.errors import ErrorCode, Issue
from planner.domain.models import ANYWHERE, BaseData, SessionState, TaskType

if TYPE_CHECKING:
    from datetime import date, datetime

    from pydantic_core import ErrorDetails

DELETE_ALL: Final = "all"
"""Value of `delete_task.task_id` that targets every task."""

_NOTHING_TO_DELETE: Final = Issue(
    ErrorCode.UNKNOWN_TASK, "task_id", "there are no tasks to delete; the list is already empty"
)

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


class _TimedTask(Protocol):
    """Anything with a start and a deadline: a new task's arguments or a stored task."""

    @property
    def earliest_start(self) -> datetime | None: ...

    @property
    def deadline(self) -> datetime | None: ...


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
        return Rejected(_issues_from_pydantic(exc.errors(), tool))
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


def _issues_from_pydantic(errors: list[ErrorDetails], tool: type[BaseModel]) -> tuple[Issue, ...]:
    """Turn pydantic errors into issues, one per error except unexpected arguments.

    Unexpected arguments are grouped per level, so the valid fields are listed once per level.
    """
    unexpected: dict[tuple[int | str, ...], list[str]] = {}
    for error in errors:
        if _code_of(error) is ErrorCode.UNEXPECTED_ARG:
            location = ".".join(str(part) for part in error["loc"])
            unexpected.setdefault(error["loc"][:-1], []).append(location)
    issues: list[Issue] = []
    for error in errors:
        code = _code_of(error)
        if code is not ErrorCode.UNEXPECTED_ARG:
            location = ".".join(str(part) for part in error["loc"])
            issues.append(Issue(code, location or "args", error["msg"]))
        elif (parent := error["loc"][:-1]) in unexpected:
            issues.append(_unexpected_issue(tool, parent, unexpected.pop(parent)))
    return tuple(issues)


def _code_of(error: ErrorDetails) -> ErrorCode:
    kind = error["type"]
    try:
        return ErrorCode(kind)  # custom errors are raised with an ErrorCode value as type
    except ValueError:
        return _CODE_BY_PYDANTIC_TYPE.get(kind, ErrorCode.BAD_TYPE)


def _unexpected_issue(
    tool: type[BaseModel], parent: tuple[int | str, ...], locations: list[str]
) -> Issue:
    # Small models guess field names; naming the valid ones lets them correct in one step.
    valid = _fields_at(tool, parent)
    if not valid:
        message = "this tool takes no arguments"
    elif len(locations) == 1:
        message = f"not a valid argument here (valid: {', '.join(valid)})"
    else:
        message = f"not valid arguments here (valid: {', '.join(valid)})"
    return Issue(ErrorCode.UNEXPECTED_ARG, ", ".join(locations), message)


def _fields_at(model: type[BaseModel], path: tuple[int | str, ...]) -> list[str]:
    """Name the fields of the (nested) model that `path` points into."""
    for part in path:
        # Only a nested model forbids extra keys, so every step of the path is a model field.
        model = cast("type[BaseModel]", model.model_fields[str(part)].annotation)
    return list(model.model_fields)


def _task_consistency(prefix: str, task_type: TaskType, fields: _TimedTask) -> list[Issue]:
    """Cross-field checks on a whole task, as it would be after the call."""
    issues = []
    if task_type is TaskType.FIXED_EVENT and fields.earliest_start is None:
        message = "a fixed event needs its start time here; ask the user for it"
        issues.append(Issue(ErrorCode.MISSING_ARG, f"{prefix}earliest_start", message))
    if (
        fields.earliest_start is not None
        and fields.deadline is not None
        and fields.deadline < fields.earliest_start
    ):
        message = "the deadline is before the earliest start"
        issues.append(Issue(ErrorCode.BAD_VALUE, f"{prefix}deadline", message))
    return issues


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
                    *_task_consistency("", args.type or TaskType.FLEXIBLE, args),
                ]
            case UpdateTaskArgs():
                return self._check_update(args)
            case DeleteTaskArgs():
                if args.task_id == DELETE_ALL:
                    return [] if self.state.tasks else [_NOTHING_TO_DELETE]
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
            case SetLocationArgs():
                return self._known_place(args.place_id, "place_id", allow_anywhere=False)
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
        current = self.state.task(args.task_id)
        if current is not None:
            merged = current.model_copy(update=patch.model_dump(exclude_none=True))
            issues += _task_consistency("fields.", merged.type, merged)
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
