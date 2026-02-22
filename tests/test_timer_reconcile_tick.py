from __future__ import annotations

from pathlib import Path

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.senses.bus import Bus
from alphonse.agent.nervous_system.senses.timer import TimerSense


def test_timer_emits_reconcile_tick_when_idle(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    apply_schema(db_path)
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.setenv("TIMER_POLL_SECONDS", "0.05")
    monkeypatch.setenv("JOB_RECONCILIATION_INTERVAL_SECONDS", "0.01")

    bus = Bus()
    timer = TimerSense()
    timer.start(bus)
    try:
        signal = bus.get(timeout=2.0)
        assert signal is not None
        assert signal.type == "timer.fired"
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        assert str(payload.get("kind") or "") == "jobs_reconcile"
    finally:
        timer.stop()

