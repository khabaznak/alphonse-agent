from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.senses.bus import Bus
from alphonse.agent.nervous_system.senses.timer import (
    MAX_ACCEPTABLE_TRIGGER_LATENCY_SECONDS,
    TimedSignalRecord,
    TimerSense,
    _allowed_lag_seconds,
)
from alphonse.agent.nervous_system.timed_store import list_timed_signals
from alphonse.agent.tools.scheduler import SchedulerTool


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def test_create_reminder_normalizes_fire_at_and_delivery_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    apply_schema(db_path)
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))

    class _FakeTimeLlm:
        def complete(self, system_prompt: str, user_prompt: str) -> str:
            _ = system_prompt
            marker = "- now_local: "
            now_line = next((line for line in user_prompt.splitlines() if line.startswith(marker)), "")
            now_local = datetime.fromisoformat(now_line.replace(marker, "", 1).strip())
            return (now_local + timedelta(minutes=1)).astimezone(timezone.utc).isoformat()

    tool = SchedulerTool(llm_client=_FakeTimeLlm())
    result = tool.create_reminder(
        for_whom="me",
        time="in 1 minute",
        message="Ir a banarme",
        timezone_name="America/Mexico_City",
        channel_target="8553589429",
        from_="telegram",
    )
    assert isinstance(result, dict)

    rows = list_timed_signals(limit=5)
    reminder = next((row for row in rows if str((row.get("payload") or {}).get("kind") or "") == "reminder"), None)
    assert isinstance(reminder, dict)
    trigger_at = str(reminder.get("trigger_at") or "")
    target = str(reminder.get("target") or "")
    payload = reminder.get("payload")
    assert isinstance(payload, dict)

    fire_at = str(payload.get("fire_at") or "")
    delivery_target = str(payload.get("delivery_target") or "")

    parsed_trigger = _parse_iso(trigger_at)
    parsed_fire = _parse_iso(fire_at)
    now = datetime.now(timezone.utc)

    assert target == "8553589429"
    assert delivery_target == "8553589429"
    assert parsed_trigger == parsed_fire
    assert now <= parsed_trigger <= now + timedelta(minutes=2)


def test_timer_dispatches_when_now_gte_fire_at(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    apply_schema(db_path)
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.setenv("TIMER_POLL_SECONDS", "0.05")

    past = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    tool = SchedulerTool()
    tool.create_reminder(
        for_whom="current_conversation",
        time=past,
        message="test due now",
        timezone_name="America/Mexico_City",
        channel_target="8553589429",
        from_="telegram",
    )

    bus = Bus()
    timer = TimerSense()
    timer.start(bus)
    try:
        signal = bus.get(timeout=2.0)
        assert signal is not None
        assert signal.type == "timed_signal.fired"
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        assert str(payload.get("kind") or "") == "reminder"
        assert str(payload.get("target") or "") == "8553589429"
    finally:
        timer.stop()


def test_allowed_lag_for_one_shot_uses_default_window() -> None:
    now = datetime.now(timezone.utc)
    record = TimedSignalRecord(
        id="s1",
        trigger_at=now,
        timezone="America/Mexico_City",
        status="pending",
        signal_type="timed_signal",
        payload={},
        target="8553589429",
        origin="tests",
        correlation_id=None,
        fired_at=now,
    )
    assert _allowed_lag_seconds(record, now) == float(MAX_ACCEPTABLE_TRIGGER_LATENCY_SECONDS)


def test_allowed_lag_for_daily_rrule_uses_catchup_window() -> None:
    now = datetime.now(timezone.utc)
    record = TimedSignalRecord(
        id="s2",
        trigger_at=now - timedelta(days=1),
        timezone="America/Mexico_City",
        status="pending",
        signal_type="timed_signal",
        payload={},
        target="8553589429",
        origin="tests",
        correlation_id=None,
        fired_at=None,
    )
    lag = _allowed_lag_seconds(record, now)
    assert lag == float(MAX_ACCEPTABLE_TRIGGER_LATENCY_SECONDS)
