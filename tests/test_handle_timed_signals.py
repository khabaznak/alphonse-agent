from __future__ import annotations

from alphonse.agent.actions.handle_timed_signals import HandleTimedSignalsAction
from alphonse.agent.nervous_system.senses.bus import Signal


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
    assert emitted.type == "api.message_received"
    payload = emitted.payload if isinstance(emitted.payload, dict) else {}
    assert str(payload.get("text") or "") == "Hi alex"


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
