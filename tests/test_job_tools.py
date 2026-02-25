from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from alphonse.agent.nervous_system.senses.bus import Signal
from alphonse.agent.services.job_runner import JobRunner
from alphonse.agent.services.job_store import JobStore
from alphonse.agent.tools.job_tools import JobCreateTool
from alphonse.agent.tools.job_tools import JobDeleteTool
from alphonse.agent.tools.job_tools import JobListTool
from alphonse.agent.tools.job_tools import JobPauseTool
from alphonse.agent.tools.job_tools import JobResumeTool
from alphonse.agent.tools.job_tools import JobRunNowTool


class _FakeBus:
    def __init__(self) -> None:
        self.events: list[Signal] = []

    def emit(self, signal: Signal) -> None:
        self.events.append(signal)


def test_job_tools_crud(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "data" / "jobs")
    runner = JobRunner(
        job_store=store,
        tick_seconds=5,
    )
    create = JobCreateTool(store)
    list_tool = JobListTool(store)
    pause = JobPauseTool(store)
    resume = JobResumeTool(store)
    delete = JobDeleteTool(store)
    run_now = JobRunNowTool(runner)

    created = create.execute(
        user_id="u1",
        name="Weekly family review",
        description="Review family tasks every week",
        schedule={
            "type": "rrule",
            "dtstart": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            "rrule": "FREQ=WEEKLY;BYDAY=TU;BYHOUR=9;BYMINUTE=0",
        },
        payload_type="internal_event",
        payload={"event_name": "family.review", "data": {"source": "job"}},
        timezone="UTC",
        safety_level="low",
    )
    assert created["status"] == "ok"
    job_id = created["result"]["job_id"]

    listed = list_tool.execute(user_id="u1")
    assert listed["status"] == "ok"
    assert any(item.get("job_id") == job_id for item in listed["result"]["jobs"])

    paused = pause.execute(user_id="u1", job_id=job_id)
    assert paused["status"] == "ok"
    assert paused["result"]["enabled"] is False

    resumed = resume.execute(user_id="u1", job_id=job_id)
    assert resumed["status"] == "ok"
    assert resumed["result"]["enabled"] is True

    run = run_now.execute(user_id="u1", job_id=job_id)
    assert run["status"] == "ok"
    assert run["result"]["execution_id"]

    removed = delete.execute(user_id="u1", job_id=job_id)
    assert removed["status"] == "ok"
    assert removed["result"]["deleted"] is True


def test_job_run_now_accepts_job_name_alias(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "data" / "jobs")
    runner = JobRunner(
        job_store=store,
        tick_seconds=5,
    )
    create = JobCreateTool(store)
    run_now = JobRunNowTool(runner)
    job_name = "Weekly family review"
    created = create.execute(
        user_id="u1",
        name=job_name,
        description="Review family tasks every week",
        schedule={
            "type": "rrule",
            "dtstart": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            "rrule": "FREQ=WEEKLY;BYDAY=TU;BYHOUR=9;BYMINUTE=0",
        },
        payload_type="internal_event",
        payload={"event_name": "family.review", "data": {"source": "job"}},
        timezone="UTC",
        safety_level="low",
    )
    assert created["status"] == "ok"
    run = run_now.execute(user_id="u1", job_name=job_name)
    assert run["status"] == "ok"
    assert run["result"]["execution_id"]


def test_job_run_now_routes_prompt_jobs_to_bus_when_present(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "data" / "jobs")
    runner = JobRunner(
        job_store=store,
        tick_seconds=5,
    )
    create = JobCreateTool(store)
    run_now = JobRunNowTool(runner)
    bus = _FakeBus()

    created = create.execute(
        user_id="u1",
        name="Daily FX update",
        description="USD to MXN at 7am",
        schedule={
            "type": "rrule",
            "dtstart": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            "rrule": "FREQ=DAILY;BYHOUR=7;BYMINUTE=0",
        },
        payload_type="prompt_to_brain",
        payload={"text": "Send Alex the current USD to MXN exchange rate over Telegram"},
        timezone="UTC",
        safety_level="low",
    )
    assert created["status"] == "ok"
    run = run_now.execute(
        user_id="u1",
        job_id=str(created["result"]["job_id"]),
        state={
            "_bus": bus,
            "channel_type": "telegram",
            "channel_target": "8553589429",
            "incoming_user_id": "u1",
            "correlation_id": "corr-1",
        },
    )
    assert run["status"] == "ok"
    assert bus.events
    emitted = bus.events[-1]
    assert emitted.type == "api.message_received"
    assert emitted.correlation_id == "corr-1"
    assert str((emitted.payload or {}).get("target") or "") == "8553589429"
