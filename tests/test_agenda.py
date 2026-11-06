from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.agenda import agenda
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def home(tmp_path):
    users = V2UserStore(":memory:")
    admin = users.onboard(display_name="Admin", users_root=tmp_path / "users")
    member = users.create_user(display_name="Member")
    users.set_timezone("America/New_York", admin.user_id)
    runtime = SimpleNamespace(user_store=users, project_store=ProjectStore(":memory:"), schedule_store=ScheduledTaskStore(":memory:"))
    return runtime, admin.user_id, member.user_id, tmp_path


def create(home, name="Routine", **kwargs):
    runtime, actor, _, _ = home
    return runtime.schedule_store.create_task(owner_user_id=actor, name=name, prompt="PRIVATE PROMPT", timezone_name="America/New_York", now=datetime(2026, 10, 1, tzinfo=timezone.utc), **kwargs)


def read(home, start="2026-10-31", end="2026-11-03", actor=None):
    return agenda(home[0], actor=actor or home[1], start_day=start, end_day=end)


def test_dst_overlaps_past_and_immutable_definitions(home):
    a = create(home, schedule_kind="rrule", rrule="FREQ=DAILY;COUNT=5", dtstart="2026-10-30T09:00:00-04:00")
    create(home, name="Overlapping", schedule_kind="once", run_at="2026-11-01T09:00:00-05:00")
    result = read(home)
    rows = [r for r in result["items"] if r["scheduled_task_id"] == a.scheduled_task_id]
    assert [r["at"] for r in rows] == ["2026-10-31T13:00:00+00:00", "2026-11-01T14:00:00+00:00", "2026-11-02T14:00:00+00:00"]
    assert len([r for r in result["items"] if r["at"] == "2026-11-01T14:00:00+00:00"]) == 2
    assert home[0].schedule_store.get_task(a.scheduled_task_id) == a
    assert "PRIVATE PROMPT" not in str(result)
    assert all(r["execution_status"] is None for r in rows)


def test_day_range_uses_civil_midnights_across_dst(home):
    create(home, name="Before end", schedule_kind="once", run_at="2026-11-02T04:59:59Z")
    create(home, name="At end", schedule_kind="once", run_at="2026-11-02T05:00:00Z")
    result = read(home, "2026-11-01", "2026-11-02")
    assert [r["name"] for r in result["items"]] == ["Before end"]  # 25-hour day


@pytest.mark.parametrize("start,end", [("2026-01-01", "2027-01-01"), ("2026-01-01", "2026-01-01"), ("bad", "2026-01-01")])
def test_rejects_unbounded_or_invalid_ranges(home, start, end):
    with pytest.raises(ValueError):
        read(home, start, end)


def test_private_other_owner_and_revoked_project(home):
    runtime, actor, other, root = home
    project = runtime.project_store.create_project(name="Private", root_path=str(root / "private"), owner_user_id=other, visibility="shared")
    runtime.project_store.add_member(project.project_id, actor)
    create(home, name="Revocable", schedule_kind="once", run_at="2026-11-01T14:00:00Z", project_id=project.project_id)
    assert len(read(home)["items"]) == 1
    runtime.project_store.remove_member(project.project_id, actor)
    assert read(home)["items"] == []
    assert read(home, actor=other)["items"] == []
    with pytest.raises(PermissionError):
        read(home, actor="missing")


def test_paused_cancelled_and_execution_are_distinct(home):
    runtime, _, _, _ = home
    task = create(home, schedule_kind="once", run_at="2026-11-01T14:00:00Z")
    runtime.schedule_store.cancel_task(task.scheduled_task_id)
    row = read(home)["items"][0]
    assert row["schedule_status"] == "cancelled" and row["execution_status"] is None
    runtime.schedule_store.record_execution(scheduled_task_id=task.scheduled_task_id, run_id="legacy", status="failed", started_at="2026-11-01T15:00:00+00:00", error="SECRET ERROR")
    result = read(home)
    assert any(r["kind"] == "execution" and r["execution_status"] == "failed" for r in result["items"])
    assert "SECRET ERROR" not in str(result)


def test_density_limit_is_explicit(home):
    create(home, schedule_kind="rrule", rrule="FREQ=MINUTELY", dtstart="2026-10-31T00:00:00-04:00")
    result = read(home)
    assert result["truncated"] and len(result["items"]) <= 500


def test_occurrence_time_survives_delayed_execution_and_no_writes(home):
    runtime = home[0]
    task = create(home, schedule_kind="once", run_at="2026-11-01T14:00:00Z")
    claims = runtime.schedule_store.claim_due_occurrences(worker_id="fixture", now=datetime(2026, 11, 4, tzinfo=timezone.utc))
    assert len(claims) == 1
    before = runtime.schedule_store.list_executions(scheduled_task_id=task.scheduled_task_id)
    result = read(home)
    row = result["items"][0]
    assert row["at"] == "2026-11-01T14:00:00+00:00"
    assert row["execution_status"] == "claimed" and row["kind"] == "occurrence"
    assert runtime.schedule_store.list_executions(scheduled_task_id=task.scheduled_task_id) == before


def test_old_secondly_schedule_is_bounded(home):
    create(home, schedule_kind="rrule", rrule="FREQ=SECONDLY", dtstart="2026-10-01T00:00:00Z")
    assert read(home)["truncated"]


def test_ipc_validates_actor_and_range(home):
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.ipc import V2DaemonServer
    daemon = object.__new__(V2Daemon)
    daemon.runtime = home[0]
    server = V2DaemonServer(daemon)
    # Exercise the public dispatcher, including missing identity.
    with pytest.raises(PermissionError):
        server._dispatch({"method": "agenda", "params": {"start_day": "2026-11-01", "end_day": "2026-11-02"}})


def test_spring_forward_is_a_23_hour_day(home):
    create(home, name="Before start", schedule_kind="once", run_at="2026-03-08T04:59:59Z")
    create(home, name="At start", schedule_kind="once", run_at="2026-03-08T05:00:00Z")
    create(home, name="Before end", schedule_kind="once", run_at="2026-03-09T03:59:59Z")
    create(home, name="At end", schedule_kind="once", run_at="2026-03-09T04:00:00Z")
    assert {r["name"] for r in read(home, "2026-03-08", "2026-03-09")["items"]} == {"At start", "Before end"}
