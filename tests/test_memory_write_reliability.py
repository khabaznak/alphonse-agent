from __future__ import annotations

from pathlib import Path
from typing import Any

from alphonse.agent.cognition.memory import service as memory_service
from alphonse.agent.cognition.memory import MemoryService
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.nervous_system import users as users_store
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.senses.bus import Signal


class _FakeBus:
    def __init__(self) -> None:
        self.events: list[Signal] = []

    def emit(self, signal: Signal) -> None:
        self.events.append(signal)


def test_memory_write_retries_then_succeeds(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "nerve-db-memory-retry-1"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    users_store.upsert_user({"user_id": "alex", "display_name": "Alex", "is_active": True})
    attempts: list[int] = []

    def _flaky_append_episode(**_: Any) -> dict[str, Any]:
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("temporary_io_failure")
        return {"ok": True}

    monkeypatch.setattr(memory_service, "append_episode", _flaky_append_episode)
    monkeypatch.setenv("ALPHONSE_MEMORY_WRITE_MAX_RETRIES", "3")
    monkeypatch.setenv("ALPHONSE_MEMORY_WRITE_RETRY_BACKOFF_SECONDS", "0")

    memory_service.record_plan_step_completion(
        state={"correlation_id": "corr-memory-retry-1", "owner_id": "alex"},
        task_record=TaskRecord(goal="test", status="running"),
        current={"step_id": "step_1", "status": "executed"},
        proposal={"kind": "call_tool"},
        correlation_id="corr-memory-retry-1",
    )

    assert len(attempts) == 3


def test_memory_write_retry_exhausted_escalates_to_admin(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "nerve-db-memory-retry-2"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    users_store.upsert_user({"user_id": "alex", "display_name": "Alex", "is_active": True})
    bus = _FakeBus()

    def _always_fail(**_: Any) -> dict[str, Any]:
        raise RuntimeError("disk_full")

    monkeypatch.setattr(memory_service, "append_episode", _always_fail)
    monkeypatch.setattr(memory_service, "_resolve_admin_telegram_target", lambda: "8553589429")
    monkeypatch.setenv("ALPHONSE_MEMORY_WRITE_MAX_RETRIES", "2")
    monkeypatch.setenv("ALPHONSE_MEMORY_WRITE_RETRY_BACKOFF_SECONDS", "0")
    memory_service._MEMORY_ESCALATION_DEDUPE.clear()

    memory_service.record_after_tool_call(
        state={"correlation_id": "corr-memory-retry-2", "_bus": bus, "owner_id": "alex", "incoming_user_id": "alex"},
        task_record=TaskRecord(goal="test", status="running"),
        current={"step_id": "step_1"},
        tool_name="communication.send_message",
        args={"To": "8553589429", "Message": "hello"},
        result={"output": {"ok": True}, "exception": None, "metadata": {}},
        correlation_id="corr-memory-retry-2",
    )

    assert bus.events
    assert bus.events[-1].type == "sense.runtime.message.user.received"
    payload = bus.events[-1].payload if isinstance(bus.events[-1].payload, dict) else {}
    assert payload.get("schema_version") == "1.0"


def test_record_after_tool_call_persists_output_summary_for_search(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHONSE_MEMORY_ROOT", str(tmp_path / "memory"))
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    users_store.upsert_user({"user_id": "alex", "display_name": "Alex", "is_active": True})
    memory_service.record_after_tool_call(
        state={"correlation_id": "corr-memory-summary-1", "owner_id": "alex", "incoming_user_id": "alex"},
        task_record=TaskRecord(goal="research potential client", status="running", task_id="task_client_1"),
        current={"step_id": "step_1"},
        tool_name="execution.call_mcp",
        args={"profile": "web", "operation": "search"},
        result={
            "output": {
                "company": "Acme Logistics",
                "hq": "Guadalajara",
                "contact": "ops@acme.example",
            },
            "exception": None,
            "metadata": {},
        },
        correlation_id="corr-memory-summary-1",
    )
    hits = MemoryService().search_episodes(user_id="alex", query="Acme Logistics")
    assert hits
