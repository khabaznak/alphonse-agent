from __future__ import annotations

import types

from alphonse.agent.actions.handle_timed_signals import HandleTimedSignalsAction
from alphonse.agent.actions.handle_timed_signals import _emit_brain_payload_to_bus
from alphonse.agent.nervous_system.senses.bus import Bus
from alphonse.agent.nervous_system.senses.bus import Signal
from alphonse.agent.tools.base import ToolDefinition
from alphonse.agent.tools.spec import ToolSpec


class _FakeBus:
    def __init__(self) -> None:
        self.events: list[Signal] = []

    def emit(self, signal: Signal) -> None:
        self.events.append(signal)


def test_conscious_reminder_dispatch_prefers_message_text_over_internal_prompt() -> None:
    action = HandleTimedSignalsAction()
    bus = _FakeBus()
    signal = Signal(
        type="timed_signal.fired",
        payload={
            "timed_signal_id": "tsig_1",
            "mind_layer": "conscious",
            "target": "8553589429",
            "payload": {
                "prompt": "Hi alex",
                "message_text": "ignored fallback",
                "agent_internal_prompt": "You just remembered to set a reminder.",
                "delivery_target": "8553589429",
            },
        },
        source="timer",
        correlation_id="corr-reminder-dispatch-text",
    )

    action.execute({"signal": signal, "ctx": bus, "state": None, "outcome": None})
    assert bus.events
    emitted = bus.events[-1]
    assert emitted.type == "timed_signal.conscious_payload"
    payload = emitted.payload if isinstance(emitted.payload, dict) else {}
    content = payload.get("content") if isinstance(payload.get("content"), dict) else {}
    assert str(content.get("text") or "") == "Hi alex"


def test_conscious_reminder_dispatch_with_real_bus_contract() -> None:
    action = HandleTimedSignalsAction()
    bus = Bus()
    signal = Signal(
        type="timed_signal.fired",
        payload={
            "timed_signal_id": "tsig_2",
            "mind_layer": "conscious",
            "target": "8553589429",
            "payload": {
                "message_text": "Take a shower now.",
                "delivery_target": "8553589429",
            },
        },
        source="timer",
        correlation_id="corr-reminder-real-bus",
    )

    result = action.execute({"signal": signal, "ctx": bus, "state": None, "outcome": None})
    assert result.intention_key == "NOOP"
    emitted = bus.get(timeout=0.1)
    assert emitted is not None
    assert emitted.type == "timed_signal.conscious_payload"


def test_timer_fired_runs_jobs_reconcile_without_dispatch(monkeypatch) -> None:
    action = HandleTimedSignalsAction()
    bus = _FakeBus()
    calls: list[bool] = []

    class _FakeReconciler:
        def reconcile(self, **_: object) -> dict[str, int]:
            calls.append(True)
            return {
                "scanned": 1,
                "updated": 1,
                "stale_removed": 0,
                "executed": 0,
                "advanced_without_run": 0,
                "failed": 0,
                "overdue_active_jobs": 0,
                "due_pending_timed_signals": 1,
            }

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_timed_signals.ScheduledJobsReconciler",
        lambda: _FakeReconciler(),
    )
    signal = Signal(
        type="timer.fired",
        payload={"kind": "jobs_reconcile"},
        source="timer",
        correlation_id="corr-reconcile-1",
    )

    result = action.execute({"signal": signal, "ctx": bus, "state": None, "outcome": None})
    assert result.intention_key == "NOOP"
    assert calls == [True]
    assert bus.events == []


def test_timer_fired_ensures_weekly_memory_maintenance_signal(monkeypatch) -> None:
    action = HandleTimedSignalsAction()
    bus = _FakeBus()
    calls: list[str] = []

    class _FakeReconciler:
        def reconcile(self, **_: object) -> dict[str, int]:
            return {
                "scanned": 0,
                "updated": 0,
                "stale_removed": 0,
                "executed": 0,
                "advanced_without_run": 0,
                "failed": 0,
                "overdue_active_jobs": 0,
                "due_pending_timed_signals": 0,
            }

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_timed_signals.ScheduledJobsReconciler",
        lambda: _FakeReconciler(),
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_timed_signals._ensure_weekly_memory_maintenance_signal",
        lambda: calls.append("ensured"),
    )

    signal = Signal(
        type="timer.fired",
        payload={"kind": "jobs_reconcile"},
        source="timer",
        correlation_id="corr-reconcile-2",
    )
    _ = action.execute({"signal": signal, "ctx": bus, "state": None, "outcome": None})
    assert calls == ["ensured"]


def test_memory_maintenance_timed_signal_runs_maintenance(monkeypatch) -> None:
    action = HandleTimedSignalsAction()
    bus = _FakeBus()
    calls: list[str] = []

    class _FakeMemoryService:
        def run_maintenance(self, **_: object) -> dict[str, int]:
            calls.append("maintenance")
            return {
                "users_scanned": 1,
                "summaries_written": 1,
                "deleted_daily": 0,
                "deleted_weekly": 0,
            }

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_timed_signals.MemoryService",
        lambda: _FakeMemoryService(),
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_timed_signals._reschedule_weekly_memory_maintenance",
        lambda **_: calls.append("rescheduled"),
    )

    signal = Signal(
        type="timed_signal.fired",
        payload={
            "timed_signal_id": "runtime.memory.weekly.maintenance",
            "payload": {"kind": "memory_maintenance"},
        },
        source="timer",
        correlation_id="corr-memory-weekly-1",
    )
    result = action.execute({"signal": signal, "ctx": bus, "state": None, "outcome": None})
    assert result.intention_key == "NOOP"
    assert calls == ["maintenance", "rescheduled"]
    assert bus.events == []


def test_job_trigger_bus_prompt_uses_payload_text_not_setup_metadata() -> None:
    bus = _FakeBus()
    _emit_brain_payload_to_bus(
        bus=bus,
        signal_payload={"target": "u1", "origin": "telegram"},
        inner={},
        user_id="u1",
        signal=types.SimpleNamespace(correlation_id="corr-job-trigger"),
        brain_payload={
            "payload_type": "prompt_to_brain",
            "job_id": "job_123",
            "payload": {
                "prompt_text": "Send voice note containing a stoic quote",
                "agent_internal_prompt": "Create scheduled job daily...",
                "source_instruction": "Create scheduled job daily...",
            },
        },
    )
    assert bus.events
    emitted = bus.events[-1]
    payload = emitted.payload if isinstance(emitted.payload, dict) else {}
    content = payload.get("content") if isinstance(payload.get("content"), dict) else {}
    text = str(content.get("text") or "")
    assert text == "Send voice note containing a stoic quote"
    assert "Create scheduled job" not in text


def test_timed_signal_executes_canonical_tool_call_payload(monkeypatch) -> None:
    called: dict[str, object] = {}

    class _FakeExecutor:
        def execute(self, *, To: str, Message: str, Channel: str | None = None, state: dict | None = None):  # noqa: N803
            called["To"] = To
            called["Message"] = Message
            called["Channel"] = Channel
            called["state"] = state
            return {"output": {"ok": True}, "exception": None}

    class _FakeRegistry:
        def get(self, key: str):
            if key != "communication.send_message":
                return None
            spec = ToolSpec(
                canonical_name="communication.send_message",
                summary="fake",
                description="fake",
                input_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                output_schema={"type": "object", "additionalProperties": True},
            )
            return ToolDefinition(spec=spec, executor=_FakeExecutor())

    monkeypatch.setattr(
        "alphonse.agent.tools.registry.build_default_tool_registry",
        lambda: _FakeRegistry(),
    )
    action = HandleTimedSignalsAction()
    result = action.execute(
        {
            "signal": Signal(
                type="timed_signal.fired",
                payload={
                    "timed_signal_id": "tsig_tool_1",
                    "target": "8553589429",
                    "origin": "telegram",
                    "payload": {
                        "tool_call": {
                            "kind": "call_tool",
                            "tool_name": "communication.send_message",
                            "args": {"To": "8553589429", "Message": "hola", "Channel": "telegram"},
                        }
                    },
                },
                source="timer",
                correlation_id="corr-tool-call",
            ),
            "ctx": _FakeBus(),
            "state": None,
            "outcome": None,
        }
    )
    assert result.intention_key == "NOOP"
    assert str(called.get("To") or "") == "8553589429"
    assert str(called.get("Message") or "") == "hola"
