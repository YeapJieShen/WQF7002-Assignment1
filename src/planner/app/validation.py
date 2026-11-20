# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
"""Validation of model tool calls before anything runs.

Two stages, both deterministic:
1. Shape: allowlist, then the tool's argument model (types, ranges, no unexpected arguments).
2. Meaning: checks that need the current state (places, task IDs, horizon, dependency cycles),
   and that a stated duration or deadline quotes the user's own words from this turn.

A rejected call carries issues phrased for the model to correct; it is never silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import TYPE_CHECKING, Final, Protocol, cast

from pydantic import BaseModel, ValidationError

from planner.app.prompt import is_forwarded
from planner.app.tools.args import (
    MAX_TASK_MIN,
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
    from collections.abc import Sequence
    from datetime import date, datetime

    from pydantic_core import ErrorDetails

DELETE_ALL: Final = "all"
"""Value of `delete_task.task_id` that targets every task."""

_WORD_VALUES: Final[dict[str, float]] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "half": 0.5, "a": 1, "an": 1,
}  # fmt: skip
_NUMBER_WORDS: Final = "|".join((*_WORD_VALUES, "half an?"))
_TIME_UNITS: Final = "minutes?|mins?|hours?|hrs?|h"
_AMOUNT_OF_TIME: Final = re.compile(
    rf"\b(?:(?P<digits>\d+(?:\.\d+)?)\s*|(?P<word>{_NUMBER_WORDS})\s+)(?P<unit>{_TIME_UNITS})\b"
)
"""A duration quote must state an amount of time: a number (digits or word) and a unit."""
_OTHER_AMOUNT: Final = re.compile(
    r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen"
    r"|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|half|quarter)\b"
)
"""A number left beside the amount ("an hour and a half", "1-2 hours") makes it not plain."""
_NOT_A_REFUSAL: Final = (
    "The daily limit is the user's own setting, not a reason to refuse: plan_day fits what it "
    "can, so report the plan, its total and what did not fit"
)
_MIN_DEADLINE_QUOTE: Final = 3
"""A deadline quote is the user's relative phrase; this and a letter or digit rule out junk."""

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


def validate_call(
    name: object,
    raw_args: object,
    state: SessionState,
    base: BaseData,
    said: Sequence[str] = (),
) -> Verdict:
    """Validate one tool call: allowlist, shape, then meaning against the current state.

    Args:
        name: Tool name as given by the model.
        raw_args: Arguments as given by the model.
        state: The current session state.
        base: The read-only reference data.
        said: The user's messages in the current turn. A duration or deadline must quote one of
            them; pasted messages (see `forwarded`) do not count.

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
        return Rejected(_issues_from_pydantic(exc.errors(), tool, raw_args))
    own = tuple(_normalised(m) for m in said if not is_forwarded(m))
    issues = _MeaningChecker(state, base, own).check(args)
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


def _issues_from_pydantic(
    errors: list[ErrorDetails], tool: type[BaseModel], raw_args: dict[str, object]
) -> tuple[Issue, ...]:
    """Turn pydantic errors into issues, one per error except unexpected arguments.

    Unexpected arguments are grouped per level, so the valid fields are listed once per level.
    A duration over the per-task maximum says how to go on, using its quote's amount.
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
            message = error["msg"]
            if error["type"] == "less_than_equal" and error["loc"][-1] == "duration_min":
                message = _too_long(error["input"], _quote_beside(raw_args, error["loc"]))
            issues.append(Issue(code, location or "args", message))
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


def _quote_beside(raw_args: dict[str, object], loc: tuple[int | str, ...]) -> object:
    """The duration_quote next to the duration_min at `loc`, as the model wrote it."""
    parent = raw_args
    for part in loc[:-1]:
        # pydantic reached the field, so every step of the path is an object.
        parent = cast("dict[str, object]", parent[str(part)])
    return parent.get("duration_quote")


def _too_long(minutes: object, quote: object) -> str:
    """Say how to go on with a duration over the per-task maximum, instead of a bare range."""
    most = (
        f"{minutes} is more than {MAX_TASK_MIN} minutes ({MAX_TASK_MIN // 60} hours), "
        "the most for one task"
    )
    stated = _minutes_in(_normalised(quote)) if isinstance(quote, str) else None
    if stated is None:
        split = f"Add longer work as several tasks of at most {MAX_TASK_MIN} minutes each"
        return f"{most}. {split}, then plan the day with plan_day. {_NOT_A_REFUSAL}"
    return f"{most}. '{quote}' is {stated} minutes: {_how_to_use(stated, quote)}"


def _share(stated: int) -> int:
    """The minutes one task holds of a stated amount: all of it, or an equal part if too long."""
    parts = -(-stated // MAX_TASK_MIN)
    return -(-stated // parts)


def _how_to_use(stated: int, quote: object) -> str:
    """Tell the model what to write for a stated amount: the number, or the parts to add."""
    if stated <= MAX_TASK_MIN:
        return f"use {stated}"
    parts = -(-stated // MAX_TASK_MIN)
    return (
        f"add it as {parts} tasks of {_share(stated)} minutes, each with duration_quote "
        f"'{quote}', then plan the day with plan_day. {_NOT_A_REFUSAL}"
    )


def _minutes_in(words: str) -> int | None:
    """The minutes a normalised quote states, or None unless it holds one plain amount."""
    found = list(_AMOUNT_OF_TIME.finditer(words))
    if len(found) != 1:
        return None
    match = found[0]
    if _OTHER_AMOUNT.search(f"{words[: match.start()]} {words[match.end() :]}"):
        return None
    digits, word = match["digits"], match["word"]
    number = float(digits) if digits else _WORD_VALUES[word.split()[0]]
    return round(number * (1 if match["unit"].startswith("m") else 60))


def _wrong_amount(minutes: int, words: str) -> int | None:
    """The amount a quote states, when `minutes` is not its share within a tenth."""
    stated = _minutes_in(words)
    if stated is None or abs(minutes - _share(stated)) * 10 <= _share(stated):
        return None  # not one plain amount, or a match
    return stated


def _normalised(text: str) -> str:
    return " ".join(text.casefold().split())


def _states(what: str, words: str) -> bool:
    """Tell whether a normalised quote could state a duration or deadline at all."""
    if what == "duration":
        return _AMOUNT_OF_TIME.search(words) is not None
    return len(words) >= _MIN_DEADLINE_QUOTE and any(c.isalnum() for c in words)


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
    said: tuple[str, ...]  # the user's own words this turn, normalised by `_normalised`

    def check(self, args: ToolArgs) -> list[Issue]:  # noqa: PLR0911 - one return per tool
        """Return every semantic problem with already well-shaped arguments."""
        match args:
            case AddTaskArgs():
                return [
                    *self._stated(args.duration_min, args.duration_quote, "duration", ""),
                    *self._stated(args.deadline, args.deadline_quote, "deadline", ""),
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
        current = self.state.task(args.task_id)
        # Resending the stored value is not a change, so it needs no quote.
        duration = (
            None if current and patch.duration_min == current.duration_min else patch.duration_min
        )
        deadline = None if current and patch.deadline == current.deadline else patch.deadline
        issues = [
            *self._known_tasks((args.task_id,), "task_id"),
            *self._stated(duration, patch.duration_quote, "duration", "fields."),
            *self._stated(deadline, patch.deadline_quote, "deadline", "fields."),
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
        if current is not None:
            merged = current.model_copy(update=patch.changes())
            issues += _task_consistency("fields.", merged.type, merged)
        return issues

    def _stated(self, value: object, quote: str | None, what: str, prefix: str) -> list[Issue]:
        """Reject a duration or deadline unless it quotes the user's own words this turn."""
        value_field = prefix + ("duration_min" if what == "duration" else what)
        quote_field = f"{prefix}{what}_quote"
        words = _normalised(quote or "")
        if value is None:
            return []
        if not words:
            problem = f"{value_field} needs {quote_field}, the user's own words that state it"
        elif not any(words in message for message in self.said):
            problem = (
                f"'{quote}' is not in the user's messages this turn. If the user stated the "
                f"{what}, copy their exact words"
            )
        elif not _states(what, words):
            problem = f"'{quote}' does not state a {what}"
        elif isinstance(value, int) and (stated := _wrong_amount(value, words)) is not None:
            message = f"'{quote}' is {stated} minutes, not {value}: {_how_to_use(stated, quote)}"
            return [Issue(ErrorCode.UNSTATED_VALUE, value_field, message)]
        else:
            return []
        advice = f"do not guess: ask the user for it, or leave {value_field} out"
        message = f"{problem}. If the user has not stated a {what}, {advice}"
        return [Issue(ErrorCode.UNSTATED_VALUE, quote_field, message)]

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
