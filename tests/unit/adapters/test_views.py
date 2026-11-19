import random
from datetime import datetime

from planner.adapters.views import (
    EXAMPLES_SHOWN,
    SAMPLE_DAYS,
    SAMPLE_MIN,
    examples,
    pace_chart,
    place_label,
    place_rows,
    stage_message,
    task_rows,
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


def test_a_rejected_check_shows_its_reason_beside_a_cross() -> None:
    record = CallRecord(
        stage="check", tool="set_location", args="{}", outcome="rejected: unknown place"
    )
    message = stage_message(record)
    assert message["metadata"]["title"] == "❌ set_location · check: rejected"
    assert message["metadata"]["log"] == "unknown place"
    assert message["metadata"]["status"] == "done"


def test_accepted_declined_and_done_stages_read_plainly() -> None:
    def title(stage: str, outcome: str) -> str:
        record = CallRecord(stage=stage, tool="delete_task", args="{}", outcome=outcome)  # type: ignore[arg-type]  # stage and outcome are plain strs; CallRecord takes Literals
        return str(stage_message(record)["metadata"]["title"])

    assert title("check", "accepted") == "✅ delete_task · check: accepted"
    assert title("confirm", "declined") == "❌ delete_task · confirm: declined"
    assert title("confirm", "approved") == "✅ delete_task · confirm: approved"
    assert title("act", "Deleted T4.") == "✅ delete_task · act: done"


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
