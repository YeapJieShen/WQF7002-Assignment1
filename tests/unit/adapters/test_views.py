import random
import re
from datetime import datetime
from typing import Any

import pytest

from planner.adapters.session import Usage
from planner.adapters.views import (
    EXAMPLES_SHOWN,
    NODE_SIZE,
    SAMPLE_DAYS,
    SAMPLE_MIN,
    THINKING_WORDS,
    USAGE_HINT,
    call_message,
    edge_labels,
    examples,
    monotone_curve,
    pace_chart,
    pace_figure,
    place_label,
    place_rows,
    places_figure,
    task_rows,
    thinking,
    travel_rows,
    usage_footer,
    usage_line,
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
    assert message["metadata"]["title"].startswith("*Tool call:* *delete_task* ")
    assert "- Check: *Accepted*" in message["content"]
    assert "- Act: *Deleted T4.*" in message["content"]
    assert message["content"].startswith("Args: `delete_task` ")
    assert "✅" not in message["metadata"]["title"]


def test_the_pending_line_rotates_its_text_and_names_a_running_tool() -> None:
    first, second = thinking(0), thinking(1)
    assert first["content"] != second["content"]
    assert thinking(len(THINKING_WORDS))["content"] == first["content"]
    assert thinking(5, "add_task")["content"].endswith("Running add_task…")
    assert "metadata" not in first  # plain reply text, not a collapsible box nested in the bubble


def test_the_usage_line_gives_seconds_and_tokens() -> None:
    assert usage_line(3.24, Usage(1234, 56)) == "1,234 In / 56 Out Tokens · 3.2 s"
    assert usage_line(0.04, None) == "0.0 s"
    assert (
        usage_line(4.5, Usage(4765, 22, cached_tokens=3600))
        == "4,765 In (3,600 Cached) / 22 Out Tokens · 4.5 s"
    )


def test_the_usage_footer_wraps_the_line_for_the_page_to_place() -> None:
    assert (
        usage_footer(3.24, Usage(1, 2))
        == f'<span class="turn-usage" title="{USAGE_HINT}">1 In / 2 Out Tokens · 3.2 s</span>'
    )


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


def test_the_pace_chart_days_are_dates_in_order(state: SessionState, base: BaseData) -> None:
    days = list(pace_chart(state, base)["day"])
    assert days == sorted(days)
    assert days[0].date() == state.now.date()


def test_the_pace_figure_is_a_filled_curve_with_a_marker_and_tick_per_day(
    state: SessionState, base: BaseData
) -> None:
    curve, points = pace_figure(state, base, "#336699").data
    figure = pace_figure(state, base, "#336699")
    assert (curve.fill, curve.line.color) == ("tozeroy", "#336699")
    assert curve.fillcolor == "rgba(51, 102, 153, 0.3)"
    assert list(figure.layout.xaxis.tickvals) == list(points.x)  # every day, the last included
    assert len(points.x) == SAMPLE_DAYS + 1
    assert curve.x[0] == points.x[0]
    assert curve.x[-1] == points.x[-1]


@pytest.mark.parametrize(
    "values", [[0, 0, 240, 240, 0], [240, 240, 120, 0, 0], [0, 100, 90, 300], [5, 5, 5], [7]]
)
def test_the_curve_never_leaves_the_range_of_neighbouring_days(values: list[int]) -> None:
    curve = monotone_curve(values, 12)
    assert [v for t, v in curve if t == int(t)] == values  # passes through every day
    for t, v in curve:
        low, high = int(t), min(int(t) + 1, len(values) - 1)
        assert min(values[low], values[high]) - 1e-9 <= v <= max(values[low], values[high]) + 1e-9


def test_the_places_graph_shows_each_places_edges_on_hover(base: BaseData) -> None:
    edges, middles, nodes = places_figure(base, "#336699").data
    assert list(nodes.text) == [p.name for p in base.places]
    assert nodes.marker.size >= NODE_SIZE
    pairs = {frozenset((leg.origin, leg.destination)) for leg in base.travel}
    assert len(middles.hovertemplate) == len(pairs)
    assert list(middles.text) == edge_labels(base)
    assert len(edge_labels(base)) == len(pairs)
    assert all(re.fullmatch(r"\d+ min", label) for label in edge_labels(base))
    assert edges.x.count(None) == len(pairs)
    assert any("drive 30 min<br>transit 55 min<extra></extra>" in t for t in middles.hovertemplate)
    assert all(t.startswith("<b>") and t.endswith("<extra></extra>") for t in middles.hovertemplate)
    assert middles.hoverlabel.bgcolor == nodes.hoverlabel.bgcolor == "#ffffff"
    home = nodes.hovertemplate[0]
    assert home.startswith(f"<b>{base.places[0].name}</b>")
    assert home.count("<br>") == sum(base.places[0].id in p for p in pairs)


def test_the_pace_chart_plans_the_sample_task_day_by_day(
    state: SessionState, base: BaseData
) -> None:
    chart = pace_chart(state, base)
    assert len(chart) == SAMPLE_DAYS + 1
    assert chart["minutes"].sum() <= SAMPLE_MIN
    tight = state.model_copy(update={"pace": Pace(daily_limit_min=120)})
    assert pace_chart(tight, base)["minutes"].iloc[0] <= 120
    assert pace_chart(tight, base)["minutes"].sum() > 120


def test_a_rejection_keeps_its_reason_outside_the_italics() -> None:
    message = call_message([record("check", "rejected: unknown place")])
    assert "- Check: *Rejected* — unknown place" in message["content"]
