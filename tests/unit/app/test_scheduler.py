# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
import itertools
from datetime import date, datetime, timedelta

import pytest

from planner.app.scheduler import DayPlan, plan_day, recommend_now
from planner.domain.models import BaseData, Pace, SessionState, Task, TaskType, Weekday

WED = date(2026, 10, 28)
THU = date(2026, 10, 29)


def _minutes(plan: DayPlan, task_id: str) -> int:
    return sum(
        (b.end - b.start) // timedelta(minutes=1) for b in plan.blocks if b.task_id == task_id
    )


def _reasons(plan: DayPlan) -> dict[str, str]:
    return {u.task_id: u.reason for u in plan.unscheduled}


def _with(state: SessionState, *tasks: Task, pace: Pace | None = None) -> SessionState:
    """Replace the tasks and start the clock on Thursday morning, so Thursday is today."""
    updated = state.with_tasks(tasks).model_copy(update={"now": datetime(2026, 10, 29, 8, 0)})
    return updated.model_copy(update={"pace": pace}) if pace else updated


def _task(task_id: str, minutes: int, **fields: object) -> Task:
    return Task.model_validate(
        {"id": task_id, "title": task_id, "type": "flexible", "duration_min": minutes, **fields}
    )


def test_today_is_planned_around_class(state: SessionState, base: BaseData) -> None:
    plan = plan_day(state, base, WED)
    ids = [b.task_id for b in plan.blocks]
    assert ids[0] == "T4"  # least slack among what can be done now
    assert plan.blocks[-1].task_id == "E1"
    assert (plan.blocks[-1].start, plan.blocks[-1].end) == (
        datetime(2026, 10, 28, 19, 0),
        datetime(2026, 10, 28, 22, 0),
    )


def test_blocks_are_ordered_and_never_overlap(state: SessionState, base: BaseData) -> None:
    for day in (WED, THU):
        blocks = plan_day(state, base, day).blocks
        assert all(a.end <= b.start for a, b in itertools.pairwise(blocks))


def test_impossible_car_service_is_reported_not_scheduled(
    state: SessionState, base: BaseData
) -> None:
    plan = plan_day(state, base, WED)
    assert "T2" not in {b.task_id for b in plan.blocks}
    assert "deadline" in _reasons(plan)["T2"]


def test_deadline_in_the_past_is_named(state: SessionState, base: BaseData) -> None:
    assert _reasons(plan_day(state, base, THU))["T4"] == "its deadline has already passed"


def test_work_stops_at_the_daily_limit(state: SessionState, base: BaseData) -> None:
    busy = _with(
        state,
        _task("T1", 700),
        _task("T2", 700),
        pace=Pace(daily_limit_min=300, distribution=0),
    )
    plan = plan_day(busy, base, THU)
    assert sum(_minutes(plan, t) for t in ("T1", "T2")) == 300
    assert "no room left today" in _reasons(plan)["T2"] + _reasons(plan)["T1"]


def test_weekday_limit_replaces_the_global_one(state: SessionState, base: BaseData) -> None:
    pace = Pace(daily_limit_min=300, weekday_limit_min={Weekday.THU: 120}, distribution=0)
    assert _minutes(plan_day(_with(state, _task("T1", 700), pace=pace), base, THU), "T1") == 120


def test_zero_limit_plans_no_work(state: SessionState, base: BaseData) -> None:
    pace = Pace(daily_limit_min=0)
    plan = plan_day(_with(state, _task("T1", 60), pace=pace), base, THU)
    assert plan.blocks == ()
    assert "T1" in _reasons(plan)


def _longest_run(plan: DayPlan, long_break_min: int) -> int:
    run = longest = 0
    previous_end = None
    for block in plan.blocks:
        if previous_end is None or block.start - previous_end >= timedelta(minutes=long_break_min):
            run = 0
        run += (block.end - block.start) // timedelta(minutes=1)
        longest = max(longest, run)
        previous_end = block.end
    return longest


def test_long_break_is_forced_after_the_consecutive_limit(
    state: SessionState, base: BaseData
) -> None:
    pace = Pace(daily_limit_min=720, distribution=0, max_consecutive_min=240, long_break_min=45)
    plan = plan_day(_with(state, _task("T1", 700), pace=pace), base, THU)
    assert _minutes(plan, "T1") == 700
    assert _longest_run(plan, 45) <= 240


def test_consecutive_limit_can_be_turned_off(state: SessionState, base: BaseData) -> None:
    pace = Pace(daily_limit_min=720, distribution=0, max_consecutive_min=None)
    plan = plan_day(_with(state, _task("T1", 700), pace=pace), base, THU)
    assert _longest_run(plan, 45) > 240


def test_errand_that_would_overrun_waits_for_a_long_break(
    state: SessionState, base: BaseData
) -> None:
    tasks = [_task(f"T{i}", 150, type="errand", place="library") for i in (1, 2)]
    pace = Pace(max_consecutive_min=240, long_break_min=45)
    plan = plan_day(_with(state, *tasks, pace=pace), base, THU)
    first, second = plan.blocks
    assert second.start - first.end >= timedelta(minutes=45)


def test_errand_longer_than_the_limit_is_still_placed(state: SessionState, base: BaseData) -> None:
    pace = Pace(daily_limit_min=720, max_consecutive_min=240)
    plan = plan_day(
        _with(state, _task("T1", 300, type="errand", place="library"), pace=pace), base, THU
    )
    assert _minutes(plan, "T1") == 300


def test_flexible_task_is_chunked(state: SessionState, base: BaseData) -> None:
    plan = plan_day(_with(state, _task("T1", 360), pace=Pace(distribution=0)), base, THU)
    chunks = [b for b in plan.blocks if b.task_id == "T1"]
    assert len(chunks) == 3
    assert all(b.end - b.start <= timedelta(minutes=120) for b in chunks)
    assert _minutes(plan, "T1") == 360
    assert chunks[1].start - chunks[0].end >= timedelta(minutes=base.config.min_break_min)


def test_distribution_spreads_work_over_days_to_the_deadline(
    state: SessionState, base: BaseData
) -> None:
    task = _task("T1", 360, deadline=datetime(2026, 10, 30, 23, 59))
    front = _with(state, task, pace=Pace(distribution=0))
    spread = _with(state, task, pace=Pace(distribution=1))
    assert _minutes(plan_day(front, base, THU), "T1") == 360
    assert _minutes(plan_day(spread, base, THU), "T1") == 180


@pytest.mark.parametrize("distribution", [0, 0.25, 0.5, 1])
def test_daily_shares_add_up_to_the_task(
    state: SessionState, base: BaseData, distribution: float
) -> None:
    task = _task("T1", 360, deadline=datetime(2026, 11, 1, 23, 59))
    start = _with(state, task, pace=Pace(daily_limit_min=720, distribution=distribution))
    days = [date(2026, 10, 29) + timedelta(days=i) for i in range(4)]
    assert sum(_minutes(plan_day(start, base, d), "T1") for d in days) == 360


def test_errand_is_not_split(state: SessionState, base: BaseData) -> None:
    errand = _task("T1", 180, type="errand", place="library")
    plan = plan_day(_with(state, errand), base, THU)
    assert [b.task_id for b in plan.blocks] == ["T1"]
    assert _minutes(plan, "T1") == 180


def test_errand_waits_for_opening_hours(state: SessionState, base: BaseData) -> None:
    plan = plan_day(_with(state, _task("T1", 60, type="errand", place="workshop")), base, THU)
    assert plan.blocks[0].start >= datetime(2026, 10, 29, 9, 0)
    assert plan.blocks[0].end <= datetime(2026, 10, 29, 18, 0)


def test_prerequisite_comes_first(state: SessionState, base: BaseData) -> None:
    tasks = (_task("T1", 60, prerequisites=("T2",), deadline=datetime(2026, 10, 29, 12, 0)),)
    plan = plan_day(_with(state, *tasks, _task("T2", 60), pace=Pace(distribution=0)), base, THU)
    order = [b.task_id for b in plan.blocks]
    assert order.index("T2") < order.index("T1")


def test_dependent_waits_when_prerequisite_is_not_planned(
    state: SessionState, base: BaseData
) -> None:
    tasks = (
        _task("T1", 60, prerequisites=("T2",)),
        _task("T2", 60, earliest_start=datetime(2026, 10, 30, 9, 0)),
    )
    plan = plan_day(_with(state, *tasks), base, THU)
    assert _reasons(plan)["T1"] == "waiting on T2"
    assert "not available until" in _reasons(plan)["T2"]


def test_hardest_first_orders_tasks_without_deadline(state: SessionState, base: BaseData) -> None:
    short, long = _task("T1", 30), _task("T2", 90)
    easy = plan_day(_with(state, short, long, pace=Pace(hardest_first=False)), base, THU)
    hard = plan_day(_with(state, short, long, pace=Pace(hardest_first=True)), base, THU)
    assert easy.blocks[0].task_id == "T1"
    assert hard.blocks[0].task_id == "T2"


def test_deadline_beats_no_deadline(state: SessionState, base: BaseData) -> None:
    tasks = (_task("T1", 30), _task("T2", 90, deadline=datetime(2026, 10, 29, 20, 0)))
    assert plan_day(_with(state, *tasks), base, THU).blocks[0].task_id == "T2"


def test_task_without_duration_is_reported(state: SessionState, base: BaseData) -> None:
    dentist = Task(id="T1", title="Dentist", type=TaskType.ERRAND)
    plan = plan_day(_with(state, dentist), base, THU)
    assert plan.blocks == ()
    assert "no duration" in _reasons(plan)["T1"]


def test_unreachable_place_is_reported(state: SessionState, base: BaseData) -> None:
    far = _task("T1", 30, type="errand", place="library")
    from_grocery = state.model_copy(update={"location": "grocery"})
    no_route = base.model_copy(
        update={
            "travel": tuple(
                leg
                for leg in base.travel
                if "library" not in {leg.origin, leg.destination}
                or "grocery" not in {leg.origin, leg.destination}
            )
        }
    )
    plan = plan_day(_with(from_grocery, far), no_route, THU)
    assert "no route" in _reasons(plan)["T1"]


def test_day_that_is_over_has_nothing_planned(state: SessionState, base: BaseData) -> None:
    late = state.model_copy(update={"now": datetime(2026, 10, 28, 23, 0)})
    plan = plan_day(late, base, WED)
    assert plan.blocks == ()
    assert "T3" in _reasons(plan)


def test_event_in_progress_blocks_the_start_of_the_day(state: SessionState, base: BaseData) -> None:
    during = state.model_copy(update={"now": datetime(2026, 10, 28, 20, 0), "location": "um"})
    plan = plan_day(during, base, WED)
    assert [b.task_id for b in plan.blocks] == ["E1"]


def test_recommend_now_picks_least_slack(state: SessionState, base: BaseData) -> None:
    rec = recommend_now(state, base)
    assert rec.task_id == "T4"
    assert rec.arrive == state.now
    assert {u.task_id for u in rec.skipped} == {"T2"}
    assert "opening hours on Wed: 09:00-18:00" in rec.skipped[0].reason


def test_recommend_now_with_nothing_feasible(state: SessionState, base: BaseData) -> None:
    only_impossible = _with(state, *(t for t in state.tasks if t.id == "T2"))
    rec = recommend_now(only_impossible, base)
    assert rec.task_id is None
    assert rec.arrive is None
    assert len(rec.skipped) == 1


def test_recommend_now_ignores_fixed_events(state: SessionState, base: BaseData) -> None:
    event = _task(
        "E9", 60, type="fixed_event", earliest_start=datetime(2026, 10, 28, 16, 0), place="um"
    )
    rec = recommend_now(_with(state, event), base)
    assert rec.task_id is None
    assert rec.skipped == ()


def test_deadline_at_midnight_gives_no_share_to_the_empty_day(
    state: SessionState, base: BaseData
) -> None:
    # Deadline Fri 00:00 means Thursday is the last usable day: it must carry the whole task.
    task = _task("T1", 240, deadline=datetime(2026, 10, 30, 0, 0))
    plan = plan_day(_with(state, task, pace=Pace(distribution=1)), base, THU)
    assert _minutes(plan, "T1") == 240


def test_deadline_before_day_start_ends_the_task_the_day_before(
    state: SessionState, base: BaseData
) -> None:
    task = _task("T1", 240, deadline=datetime(2026, 10, 30, 7, 0))
    plan = plan_day(_with(state, task, pace=Pace(distribution=1)), base, THU)
    assert _minutes(plan, "T1") == 240


def test_deadline_already_passed_does_not_break_the_share(
    state: SessionState, base: BaseData
) -> None:
    task = _task("T1", 240, deadline=datetime(2026, 10, 28, 12, 0))
    plan = plan_day(_with(state, task, pace=Pace(distribution=1)), base, THU)
    assert _reasons(plan)["T1"] == "its deadline has already passed"


def test_fixed_event_without_a_start_is_reported(state: SessionState, base: BaseData) -> None:
    exam = Task(id="T1", title="Exam", type=TaskType.FIXED_EVENT, duration_min=60)
    plan = plan_day(_with(state, exam), base, THU)
    assert plan.blocks == ()
    assert "no start time" in _reasons(plan)["T1"]
