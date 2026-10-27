from planner.app.prompt import session_snapshot
from planner.domain.models import BaseData, SessionState


def test_snapshot_shows_the_session(state: SessionState, base: BaseData) -> None:
    snapshot = session_snapshot(state, base)
    assert snapshot.startswith("Current session:")
    assert "Wednesday 2026-10-28 15:40" in snapshot
    assert f"({state.location})" in snapshot
    assert f"{state.pace.daily_limit_min} minutes" in snapshot
    assert f"{base.config.horizon_days} days ahead" in snapshot
    for place in base.places:
        assert f"- {place.id}: {place.name}" in snapshot
    for task in state.tasks:
        assert f"- {task.id} {task.title}" in snapshot


def test_snapshot_marks_unknown_duration_and_no_tasks(state: SessionState, base: BaseData) -> None:
    unknown = state.tasks[0].model_copy(update={"duration_min": None})
    assert "duration unknown" in session_snapshot(state.with_tasks((unknown,)), base)
    assert "Tasks:\n- none" in session_snapshot(state.with_tasks(()), base)


def test_snapshot_falls_back_to_the_raw_location_id(state: SessionState, base: BaseData) -> None:
    lost = SessionState(now=state.now, location="mars")
    assert "mars (mars)" in session_snapshot(lost, base)
