"""Run the challenge cases against a live model and record what happened.

`python -m planner.adapters.evals evals/cases.toml --out evals/results` writes a JSON and a
Markdown report for the configured backend. Each case starts a fresh session from the sample
persona, so cases do not affect each other. Needs a running backend; CI never runs it.

The automatic verdict only checks the calls (expected tools reached validation, forbidden
tools never ran). Whether the reply is right is judged by a person reading the report.
"""

from __future__ import annotations

import argparse
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from planner.adapters.backends import build_backend
from planner.adapters.config import Settings
from planner.adapters.data import load_base, load_persona, load_rules
from planner.adapters.llm import BACKEND_ERRORS
from planner.adapters.session import Session
from planner.adapters.ui import PERSONA, forwarded
from planner.app.calls import CallRecord

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from planner.domain.models import SessionState


def default_session(state: SessionState | None = None) -> Session:
    """Start a conversation with the configured model, from `state` or the sample persona."""
    start = state or load_persona(PERSONA)
    settings = Settings()
    return Session(
        build_backend(settings).model(None),
        load_base(),
        load_rules(),
        start,
        max_rejections=settings.max_rejections,
    )


class Case(BaseModel):
    """One scripted user turn and what should happen.

    Attributes:
        id: Short name, such as "1-now".
        kind: The test category from the assignment.
        input: What the user types or pastes.
        pasted: Send it as a forwarded message, as the UI does for Ctrl+V.
        answer: How to answer a confirmation if one is asked; None leaves it pending.
        expect_tools: Tools the model should call (checked at the validation stage).
        forbid_acts: Tools that must never run.
        expected: The behaviour a person checks the reply against.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: Literal["normal", "unknown", "injection", "unsafe", "backup"]
    input: str
    pasted: bool = False
    answer: bool | None = None
    expect_tools: tuple[str, ...] = ()
    forbid_acts: tuple[str, ...] = ()
    expected: str


class Outcome(BaseModel):
    """What one case produced."""

    model_config = ConfigDict(frozen=True)

    case: Case
    reply: str
    confirm: str | None
    calls: list[CallRecord]
    tasks_before: int
    tasks_after: int
    seconds: float
    missing: list[str]
    forbidden: list[str]

    @property
    def calls_ok(self) -> bool:
        """Expected tools were called and no forbidden tool ran."""
        return not self.missing and not self.forbidden


def load_cases(path: Path) -> list[Case]:
    """Read the cases from a TOML file with a `[[case]]` table per case."""
    data = tomllib.loads(path.read_text("utf-8"))
    return [Case.model_validate(c) for c in data["case"]]


def run_case(case: Case, make_session: Callable[[], Session]) -> Outcome:
    """Play one case in a fresh session and check its calls."""
    session = make_session()
    before = len(session.state.tasks)
    started = time.perf_counter()
    confirm = None
    try:
        reply = session.send(forwarded(case.input) if case.pasted else case.input)
        confirm = reply.confirm
        if confirm is not None and case.answer is not None:
            reply = session.resume(approved=case.answer)
        text = reply.text
    except BACKEND_ERRORS as exc:
        # Record the failure and move on; one slow case should not end the run.
        text = f"(backend error: {type(exc).__name__})"
    seconds = time.perf_counter() - started
    calls = session.call_log
    checked = {c.tool for c in calls if c.stage == "check"}
    ran = {c.tool for c in calls if c.stage == "act"}
    return Outcome(
        case=case,
        reply=text,
        confirm=confirm,
        calls=calls,
        tasks_before=before,
        tasks_after=len(session.state.tasks),
        seconds=round(seconds, 1),
        missing=[t for t in case.expect_tools if t not in checked],
        forbidden=[t for t in case.forbid_acts if t in ran],
    )


def to_markdown(outcomes: Sequence[Outcome], backend: str) -> str:
    """Render the run as a report: a summary table, then each case in full."""
    lines = [
        f"# Eval run: {backend}",
        "",
        "| Case | Kind | Calls OK | Seconds | Tools called |",
        "|---|---|---|---|---|",
    ]
    for o in outcomes:
        tools = ", ".join(f"{c.tool} ({c.stage})" for c in o.calls) or "none"
        mark = "yes" if o.calls_ok else "**no**"
        lines.append(f"| {o.case.id} | {o.case.kind} | {mark} | {o.seconds} | {tools} |")
    for o in outcomes:
        lines += ["", f"## {o.case.id} ({o.case.kind})", "", f"**Input:** {o.case.input}", ""]
        lines += [f"**Expected:** {o.case.expected}", "", "| Stage | Tool | Args | Outcome |"]
        lines.append("|---|---|---|---|")
        lines += [f"| {c.stage} | {c.tool} | `{c.args}` | {_cell(c.outcome)} |" for c in o.calls]
        if o.confirm:
            lines += ["", f"**Confirmation asked:** {_cell(o.confirm)}"]
        lines += ["", f"**Reply:** {_cell(o.reply)}", ""]
        lines.append(f"Tasks before/after: {o.tasks_before}/{o.tasks_after}.")
        if not o.calls_ok:
            lines.append(f"Missing calls: {o.missing}. Forbidden calls that ran: {o.forbidden}.")
    return "\n".join(lines) + "\n"


def backend_label(settings: Settings) -> str:
    """Name the configured backends, best first."""
    if settings.backend == "zerogpu":
        return f"zerogpu ({settings.zerogpu_model_id})"
    return " -> ".join(f"{e.name} ({e.model})" for e in settings.llm_endpoints)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - needs a live backend
    """Run every case (or the ones named) and write the reports."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", type=Path)
    parser.add_argument("--out", type=Path, default=Path("evals/results"))
    parser.add_argument("--only", nargs="*", help="case ids to run")
    args = parser.parse_args(argv)
    cases = [c for c in load_cases(args.cases) if not args.only or c.id in args.only]
    backend = backend_label(Settings())
    outcomes = []
    for case in cases:
        outcome = run_case(case, default_session)
        print(f"{case.id}: calls ok={outcome.calls_ok} in {outcome.seconds}s")  # noqa: T201
        outcomes.append(outcome)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"{datetime.now():%Y%m%d-%H%M%S}"  # noqa: DTZ005 - local file name
    stem.with_suffix(".md").write_text(to_markdown(outcomes, backend), "utf-8")
    dump = "[" + ",\n".join(o.model_dump_json(indent=2) for o in outcomes) + "]\n"
    stem.with_suffix(".json").write_text(dump, "utf-8")
    print(f"wrote {stem}.md and {stem}.json")  # noqa: T201


if __name__ == "__main__":  # pragma: no cover
    main()
