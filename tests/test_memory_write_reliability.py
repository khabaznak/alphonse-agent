from __future__ import annotations

from typing import Any

from alphonse.agent.cognition.memory import service as memory_service
from alphonse.agent.nervous_system.senses.bus import Signal


class _FakeBus:
    def __init__(self) -> None:
        self.events: list[Signal] = []

    def emit(self, signal: Signal) -> None:
        self.events.append(signal)


def test_memory_write_retries_then_succeeds(monkeypatch) -> None:
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
        state={"correlation_id": "corr-memory-retry-1"},
        task_state={"goal": "test", "status": "running"},
        current={"step_id": "step_1", "status": "executed"},
        proposal={"kind": "call_tool"},
        correlation_id="corr-memory-retry-1",
    )

    assert len(attempts) == 3


def test_memory_write_retry_exhausted_escalates_to_admin(monkeypatch) -> None:
    bus = _FakeBus()

    def _always_fail(**_: Any) -> dict[str, Any]:
        raise RuntimeError("disk_full")

    monkeypatch.setattr(memory_service, "append_episode", _always_fail)
    monkeypatch.setattr(memory_service, "_resolve_admin_telegram_target", lambda: "8553589429")
    monkeypatch.setenv("ALPHONSE_MEMORY_WRITE_MAX_RETRIES", "2")
    monkeypatch.setenv("ALPHONSE_MEMORY_WRITE_RETRY_BACKOFF_SECONDS", "0")
    memory_service._MEMORY_ESCALATION_DEDUPE.clear()

    memory_service.record_after_tool_call(
        state={"correlation_id": "corr-memory-retry-2", "_bus": bus, "incoming_user_id": "alex"},
        task_state={"goal": "test", "status": "running"},
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
