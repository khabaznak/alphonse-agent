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
    bus = _FakeBus()

    created = create.execute(
        user_id="u1",
        name="Weekly family review",
        description="Review family tasks every week",
        schedule={
            "type": "rrule",
            "dtstart": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            "rrule": "FREQ=WEEKLY;BYDAY=TU;BYHOUR=9;BYMINUTE=0",
        },
        payload_type="prompt_to_brain",
        payload={"prompt_text": "Review family tasks and send a concise summary."},
        timezone="UTC",
        safety_level="low",
    )
    assert created["exception"] is None
    job_id = created["output"]["job_id"]

    listed = list_tool.execute(user_id="u1")
    assert listed["exception"] is None
    assert any(item.get("job_id") == job_id for item in listed["output"]["jobs"])

    paused = pause.execute(user_id="u1", job_id=job_id)
    assert paused["exception"] is None
    assert paused["output"]["enabled"] is False

    resumed = resume.execute(user_id="u1", job_id=job_id)
    assert resumed["exception"] is None
    assert resumed["output"]["enabled"] is True

    run = run_now.execute(
        user_id="u1",
        job_id=job_id,
        state={
            "_bus": bus,
            "channel_type": "api",
            "channel_target": "u1",
            "incoming_user_id": "u1",
            "correlation_id": "corr-crud",
        },
    )
    assert run["exception"] is None
    assert run["output"]["execution_id"]

    removed = delete.execute(user_id="u1", job_id=job_id)
    assert removed["exception"] is None
    assert removed["output"]["deleted"] is True


def test_job_run_now_accepts_job_name_alias(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "data" / "jobs")
    runner = JobRunner(
        job_store=store,
        tick_seconds=5,
    )
    create = JobCreateTool(store)
    run_now = JobRunNowTool(runner)
    bus = _FakeBus()
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
        payload_type="prompt_to_brain",
        payload={"prompt_text": "Review family tasks and send a concise summary."},
        timezone="UTC",
        safety_level="low",
    )
    assert created["exception"] is None
    run = run_now.execute(
        user_id="u1",
        job_name=job_name,
        state={
            "_bus": bus,
            "channel_type": "api",
            "channel_target": "u1",
            "incoming_user_id": "u1",
            "correlation_id": "corr-name-alias",
        },
    )
    assert run["exception"] is None
    assert run["output"]["execution_id"]


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
    assert created["exception"] is None
    run = run_now.execute(
        user_id="u1",
        job_id=str(created["output"]["job_id"]),
        state={
            "_bus": bus,
            "channel_type": "telegram",
            "channel_target": "8553589429",
            "incoming_user_id": "u1",
            "correlation_id": "corr-1",
        },
    )
    assert run["exception"] is None
    assert bus.events
    emitted = bus.events[-1]
    assert emitted.type == "sense.api.message.user.received"
    assert emitted.correlation_id == "corr-1"
    payload = emitted.payload or {}
    channel = payload.get("channel") if isinstance(payload.get("channel"), dict) else {}
    content = payload.get("content") if isinstance(payload.get("content"), dict) else {}
    controls = payload.get("controls") if isinstance(payload.get("controls"), dict) else {}
    assert str(channel.get("target") or "") == "8553589429"
    assert str(content.get("text") or "") == "Send Alex the current USD to MXN exchange rate over Telegram"
    assert bool(controls.get("force_new_task")) is True
    assert "Create scheduled job" not in str(content.get("text") or "")


def test_job_run_now_fails_when_brain_sink_missing(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "data" / "jobs")
    runner = JobRunner(
        job_store=store,
        tick_seconds=5,
    )
    create = JobCreateTool(store)
    run_now = JobRunNowTool(runner)

    created = create.execute(
        user_id="u1",
        name="Daily reminder",
        description="No bus should fail run_now",
        schedule={
            "type": "rrule",
            "dtstart": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            "rrule": "FREQ=DAILY;BYHOUR=7;BYMINUTE=0",
        },
        payload_type="prompt_to_brain",
        payload={"prompt_text": "Send the update"},
        timezone="UTC",
        safety_level="low",
    )
    assert created["exception"] is None
    run = run_now.execute(
        user_id="u1",
        job_id=str(created["output"]["job_id"]),
    )
    assert run["exception"] is not None
    assert str((run["exception"] or {}).get("code") or "") == "job_run_failed"


def test_job_create_rejects_prompt_payload_without_execution_text(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "data" / "jobs")
    create = JobCreateTool(store)
    created = create.execute(
        user_id="u1",
        name="Daily stoic voice note",
        description="send stoic quote",
        schedule={
            "type": "rrule",
            "dtstart": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            "rrule": "FREQ=DAILY;BYHOUR=5;BYMINUTE=30",
        },
        payload_type="prompt_to_brain",
        payload={},
        timezone="UTC",
        safety_level="low",
    )
    assert created["exception"] is not None
    assert str((created.get("exception") or {}).get("code") or "") == "missing_prompt_text"


def test_job_create_normalizes_prompt_text_from_message_key(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "data" / "jobs")
    create = JobCreateTool(store)
    created = create.execute(
        user_id="u1",
        name="Daily stoic voice note",
        description="send stoic quote",
        schedule={
            "type": "rrule",
            "dtstart": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            "rrule": "FREQ=DAILY;BYHOUR=5;BYMINUTE=30",
        },
        payload_type="prompt_to_brain",
        payload={"message": "Send a voice note containing a stoic quote"},
        timezone="UTC",
        safety_level="low",
    )
    assert created["exception"] is None
    job = store.get_job(user_id="u1", job_id=str(created["output"]["job_id"]))
    assert str(job.payload.get("prompt_text") or "") == "Send a voice note containing a stoic quote"
    assert str(job.payload.get("message") or "") == "Send a voice note containing a stoic quote"


def test_job_create_requires_canonical_tool_call_payload(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "data" / "jobs")
    create = JobCreateTool(store)
    created = create.execute(
        user_id="u1",
        name="Send text",
        description="legacy payload should fail",
        schedule={
            "type": "rrule",
            "dtstart": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            "rrule": "FREQ=DAILY;BYHOUR=5;BYMINUTE=30",
        },
        payload_type="tool_call",
        payload={"tool_name": "communication.send_message", "args": {"To": "u1", "Message": "hi"}},
        timezone="UTC",
        safety_level="low",
    )
    assert created["exception"] is not None
    assert str((created.get("exception") or {}).get("code") or "") == "jobs_conscious_only_payload_type"
