from __future__ import annotations

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.registry import ActionRegistry
from alphonse.agent.cognition.intentions.intent_pipeline import IntentPipeline
from alphonse.agent.nervous_system.senses.bus import Bus
from alphonse.agent.nervous_system.senses.bus import Signal


class _NoopCoordinator:
    def deliver(self, _result, _context):  # noqa: ANN001
        return None


class _ExplodingAction(Action):
    key = "explode"

    def execute(self, context: dict):  # noqa: ANN201
        _ = context
        raise RuntimeError("boom")


def test_subconscious_failure_escalates_to_runtime_conscious_signal(monkeypatch) -> None:
    bus = Bus()
    actions = ActionRegistry()
    actions.register("explode", lambda _ctx: _ExplodingAction())
    pipeline = IntentPipeline(actions=actions, bus=bus, coordinator=_NoopCoordinator())

    monkeypatch.setattr(
        "alphonse.agent.cognition.intentions.intent_pipeline._resolve_admin_telegram_target",
        lambda: "8553589429",
    )
    monkeypatch.setattr(
        "alphonse.agent.cognition.intentions.intent_pipeline.write_trace",
        lambda _payload: None,
    )

    ctx = {
        "signal": Signal(
            type="runtime.health_check",
            source="system",
            payload={"correlation_id": "corr-runtime-1"},
            correlation_id="corr-runtime-1",
        )
    }
    pipeline.handle("explode", ctx)

    first = bus.get(timeout=0.1)
    second = bus.get(timeout=0.1)
    assert first is not None
    assert first.type == "sense.runtime.message.user.received"
    assert isinstance(first.payload, dict)
    assert first.payload.get("schema_version") == "1.0"
    assert second is not None
    assert second.type == "action.failed"


def test_conscious_failure_does_not_emit_runtime_escalation(monkeypatch) -> None:
    bus = Bus()
    actions = ActionRegistry()
    actions.register("handle_conscious_message", lambda _ctx: _ExplodingAction())
    pipeline = IntentPipeline(actions=actions, bus=bus, coordinator=_NoopCoordinator())

    monkeypatch.setattr(
        "alphonse.agent.cognition.intentions.intent_pipeline._resolve_admin_telegram_target",
        lambda: "8553589429",
    )
    monkeypatch.setattr(
        "alphonse.agent.cognition.intentions.intent_pipeline.write_trace",
        lambda _payload: None,
    )

    ctx = {
        "signal": Signal(
            type="sense.telegram.message.user.received",
            source="telegram",
            payload={"correlation_id": "corr-runtime-2"},
            correlation_id="corr-runtime-2",
        )
    }
    pipeline.handle("handle_conscious_message", ctx)

    first = bus.get(timeout=0.1)
    second = bus.get(timeout=0.1)
    assert first is not None
    assert first.type == "action.failed"
    assert second is None
