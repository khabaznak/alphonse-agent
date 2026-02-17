from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from alphonse.agent.services.job_models import JobSpec
from alphonse.agent.services.job_runner import JobRunner, route_job
from alphonse.agent.services.job_store import JobStore, compute_next_run_at
from alphonse.agent.services.scratchpad_service import ScratchpadService


class _FakeRegistry:
    def __init__(self) -> None:
        self.called = False

    def get(self, key: str):
        if key != "dummy_tool":
            return None

        registry = self

        class _Tool:
            def execute(self, *, value: str, state: dict | None = None):
                _ = state
                registry.called = True
                return {"status": "ok", "result": {"echo": value}, "error": None, "metadata": {"tool": "dummy_tool"}}

        return _Tool()


def test_rrule_next_run_basic() -> None:
    after = datetime(2026, 2, 17, 15, 0, tzinfo=timezone.utc)
    next_run = compute_next_run_at(
        schedule={
            "type": "rrule",
            "dtstart": "2026-02-17T09:00:00-06:00",
            "rrule": "FREQ=WEEKLY;BYDAY=TU;BYHOUR=9;BYMINUTE=0",
        },
        timezone_name="America/Mexico_City",
        after=after,
    )
    assert isinstance(next_run, str)
    assert next_run


def test_job_store_create_and_persist(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "data" / "jobs")
    created = store.create_job(
        user_id="u1",
        payload={
            "name": "Weekly chores",
            "description": "Family chores reminder",
            "schedule": {
                "type": "rrule",
                "dtstart": "2026-02-17T09:00:00-06:00",
                "rrule": "FREQ=WEEKLY;BYDAY=TU;BYHOUR=9;BYMINUTE=0",
            },
            "payload_type": "prompt_to_brain",
            "payload": {"prompt_text": "Prepare chores plan"},
            "timezone": "America/Mexico_City",
        },
    )
    again = store.get_job(user_id="u1", job_id=created.job_id)
    assert again.job_id == created.job_id
    assert again.next_run_at
    assert (tmp_path / "data" / "jobs" / "u1" / "jobs.json").exists()


def test_runner_executes_due_job(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "data" / "jobs")
    scratchpad = ScratchpadService(root=tmp_path / "data" / "scratchpad")
    registry = _FakeRegistry()
    runner = JobRunner(
        job_store=store,
        scratchpad_service=scratchpad,
        tool_registry=registry,
        tick_seconds=5,
    )
    created = store.create_job(
        user_id="u1",
        payload={
            "name": "Immediate tool job",
            "description": "Run deterministic tool",
            "schedule": {
                "type": "rrule",
                "dtstart": (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat(),
                "rrule": "FREQ=DAILY;INTERVAL=1",
            },
            "payload_type": "tool_call",
            "payload": {"tool_key": "dummy_tool", "args": {"value": "ok"}},
            "timezone": "UTC",
            "safety_level": "low",
            "requires_confirmation": False,
        },
    )
    outcome = runner.run_job_now(user_id="u1", job_id=created.job_id)
    assert outcome["status"] == "ok"
    assert registry.called is True
    executions = store.list_executions(user_id="u1", job_id=created.job_id, limit=5)
    assert executions
    logs = scratchpad.list_docs(user_id="u1", scope="daily", tag="jobs_log", limit=5)["docs"]
    assert logs


def test_router_sends_to_brain_for_high_safety() -> None:
    job = JobSpec(
        job_id="job_x",
        name="High risk",
        description="Needs confirmation",
        enabled=True,
        schedule={"type": "rrule", "dtstart": "2026-02-17T09:00:00+00:00", "rrule": "FREQ=DAILY"},
        timezone="UTC",
        payload_type="tool_call",
        payload={"tool_key": "dummy_tool", "args": {}},
        safety_level="high",
        requires_confirmation=False,
    )
    decision = route_job(job=job, auto_execute_high_risk=False)
    assert decision.route == "brain"

