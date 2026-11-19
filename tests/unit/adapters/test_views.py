import random
import re
from datetime import datetime
from typing import Any

from planner.adapters.views import (
    EXAMPLES_SHOWN,
    SAMPLE_DAYS,
    SAMPLE_MIN,
    THINKING_WORDS,
    call_message,
    examples,
    pace_chart,
    place_label,
    place_rows,
    task_rows,
    thinking,
    travel_rows,
)
from planner.app.calls import CallRecord
from planner.domain.models import BaseData, Pace, SessionState


def test_task_rows_describe_each_task(state: SessionState) -> None:
    rows = task_rows(state)
    assert [r[0] for r in rows] == [t.id for t in state.tasks]
    assert all(len(r) == 7 for r in rows)


def test_places_and_travel_are_tables_of_text(base: BaseData) -> None:
    assert {r[1] for r in place_rows(base)} == base.place_ids
    assert any("mon" in r[3] for r in place_rows(base))
    assert any(r[3] == "always" for r in place_rows(base))
    assert len(travel_rows(base)) == len(base.travel)


def test_the_location_chip_names_the_place_or_falls_back_to_its_id(base: BaseData) -> None:
    assert place_label(base, "office") == "📍 Office"
    assert place_label(base, "mars") == "📍 mars"


def record(stage: str, outcome: str, tool: str = "delete_task") -> CallRecord:
    return CallRecord(stage=stage, tool=tool, args="{}", outcome=outcome)  # type: ignore[arg-type]  # stage is a plain str; CallRecord takes a Literal


def badge(message: dict[str, Any]) -> tuple[str, str]:
    """The state class and text of a call row's badge."""
    found = re.search(r'badge (\w+)">([^<]+)<', message["metadata"]["title"])
    assert found
    return found.group(1), found.group(2)


def test_a_call_is_one_row_that_moves_through_its_states() -> None:
    check = record("check", "accepted")
    assert badge(call_message([check])) == ("run", "checked")
    assert call_message([check])["metadata"]["status"] == "pending"
    waiting = call_message([check], waiting=True)
    assert badge(waiting) == ("wait", "waiting for you")
    assert waiting["metadata"]["status"] == "pending"
    approved = record("confirm", "approved")
    assert badge(call_message([check, approved])) == ("run", "running")
    done = call_message([check, approved, record("act", "Deleted T4.")])
    assert badge(done) == ("ok", "done")
    assert done["metadata"]["status"] == "done"


def test_a_refused_call_ends_as_rejected_or_declined() -> None:
    assert badge(call_message([record("check", "rejected: unknown place")])) == ("bad", "rejected")
    declined = call_message([record("check", "accepted"), record("confirm", "declined")])
    assert badge(declined) == ("bad", "declined")
    assert declined["metadata"]["status"] == "done"


def test_the_row_body_lists_every_stage_and_the_title_names_the_tool() -> None:
    message = call_message([record("check", "accepted"), record("act", "Deleted T4.")])
    assert message["metadata"]["title"].startswith("delete_task ")
    assert "- check: accepted" in message["content"]
    assert "- act: Deleted T4." in message["content"]
    assert "✅" not in message["metadata"]["title"]


def test_the_pending_line_rotates_its_text_and_names_a_running_tool() -> None:
    first, second = thinking(0), thinking(1)
    assert first["content"] != second["content"]
    assert thinking(len(THINKING_WORDS))["content"] == first["content"]
    assert thinking(5, "add_task")["content"].endswith("Running add_task…")
    assert "metadata" not in first  # plain reply text, not a collapsible box nested in the bubble


def test_examples_are_three_and_match_the_loaded_data(state: SessionState, base: BaseData) -> None:
    picked = examples(state, base, random.Random(3))
    assert len(picked) == EXAMPLES_SHOWN
    assert len({e["text"] for e in picked}) == EXAMPLES_SHOWN
    every = {e["text"] for seed in range(40) for e in examples(state, base, random.Random(seed))}
    assert any(t.startswith(("Mark T", "Can I start T")) for t in every)
    assert not any(t.startswith("Add") for t in every)


def test_examples_for_an_empty_session_offer_adding_tasks(base: BaseData) -> None:
    empty = SessionState(now=datetime(2026, 11, 19, 9, 0), location="home")
    every = {e["text"] for seed in range(40) for e in examples(empty, base, random.Random(seed))}
    assert any(t.startswith("Add") for t in every)
    assert not any(t.startswith("Mark") for t in every)


def test_the_pace_chart_plans_the_sample_task_day_by_day(
    state: SessionState, base: BaseData
) -> None:
    chart = pace_chart(state, base)
    assert len(chart) == SAMPLE_DAYS + 1
    assert chart["minutes"].sum() <= SAMPLE_MIN
    tight = state.model_copy(update={"pace": Pace(daily_limit_min=120)})
    assert pace_chart(tight, base)["minutes"].iloc[0] <= 120
    assert pace_chart(tight, base)["minutes"].sum() > 120
