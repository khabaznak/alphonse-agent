from __future__ import annotations

from alphonse.agent.actions.handle_timer_fired import HandleTimerFiredAction
from alphonse.agent.nervous_system.senses.bus import Signal


class _FakeBus:
    def __init__(self) -> None:
        self.events: list[Signal] = []

    def emit(self, signal: Signal) -> None:
        self.events.append(signal)


def test_conscious_reminder_dispatch_prefers_message_text_over_internal_prompt() -> None:
    action = HandleTimerFiredAction()
    bus = _FakeBus()
    signal = Signal(
        type="timed_signal.fired",
        payload={
            "timed_signal_id": "tsig_1",
            "kind": "reminder",
            "mind_layer": "conscious",
            "dispatch_mode": "graph",
            "target": "8553589429",
            "payload": {
                "kind": "reminder",
                "message_text": "Hi alex",
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
