from __future__ import annotations

from pathlib import Path

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.timed_store import insert_timed_signal, list_timed_signals
from alphonse.agent.services.automation_payload_migrations import migrate_timed_signal_tool_call_payloads


def test_migrate_timed_signal_tool_call_payloads(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    insert_timed_signal(
        trigger_at="2026-03-28T10:00:00+00:00",
        timezone="UTC",
        payload={"payload_type": "tool_call", "tool_name": "communication.send_message", "args": {"To": "u1", "Message": "hola"}},
        target="u1",
        origin="tests",
        correlation_id="corr-1",
    )
    dry = migrate_timed_signal_tool_call_payloads(dry_run=True)
    assert int(dry.get("updated") or 0) == 1
    applied = migrate_timed_signal_tool_call_payloads(dry_run=False)
    assert int(applied.get("updated") or 0) == 1
    rows = list_timed_signals(limit=10)
    payload = rows[0].get("payload") if rows else {}
    assert isinstance(payload, dict)
    tool_call = payload.get("tool_call")
    assert isinstance(tool_call, dict)
    assert str(tool_call.get("tool_name") or "") == "communication.send_message"
