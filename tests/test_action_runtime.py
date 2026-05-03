from __future__ import annotations

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.actions.registry import ActionRegistry
from alphonse.agent.actions.runtime import ActionExecutionRuntime
from alphonse.agent.io import NormalizedOutboundMessage
from alphonse.agent.nervous_system.senses.bus import Bus, Signal


class _ExplodingAction(Action):
    key = "explode"

    def execute(self, context: dict):  # noqa: ANN201
        _ = context
        raise RuntimeError("boom")


class _MessageAction(Action):
    key = "message"

    def execute(self, context: dict) -> ActionResult:
        _ = context
        return ActionResult(
            intention_key="MESSAGE_READY",
            payload={"message": "hello"},
            urgency="normal",
            delivers_message=True,
        )


class _NoopCoordinator:
    def deliver(self, _result, _context):  # noqa: ANN001
        return None


class _FakeCoordinator:
    def deliver(self, _result, _context):  # noqa: ANN001
        return NormalizedOutboundMessage(
            message="hello",
            channel_type="cli",
            channel_target="cli",
            audience={"kind": "system", "id": "system"},
            correlation_id="corr-message",
        )


class _FakeAdapter:
    def __init__(self) -> None:
        self.deliveries: list[NormalizedOutboundMessage] = []

    def deliver(self, message: NormalizedOutboundMessage) -> None:
        self.deliveries.append(message)


class _FakeRegistry:
    def __init__(self, adapter: _FakeAdapter) -> None:
        self._adapter = adapter

    def get_extremity(self, channel_type: str):  # noqa: ANN201
        if channel_type == "cli":
            return self._adapter
        return None


def test_subconscious_failure_escalates_to_runtime_conscious_signal(monkeypatch) -> None:
    bus = Bus()
    actions = ActionRegistry()
    actions.register("explode", lambda _ctx: _ExplodingAction())
    runtime = ActionExecutionRuntime(actions=actions, bus=bus, coordinator=_NoopCoordinator())

    monkeypatch.setattr(
        "alphonse.agent.actions.runtime._resolve_admin_telegram_target",
        lambda: "8553589429",
    )

    ctx = {
        "signal": Signal(
            type="runtime.health_check",
            source="system",
            payload={"correlation_id": "corr-runtime-1"},
            correlation_id="corr-runtime-1",
        )
    }

    try:
        runtime.execute("explode", ctx)
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected runtime.execute to re-raise action failure")

    first = bus.get(timeout=0.1)
    second = bus.get(timeout=0.1)
    assert first is not None
    assert first.type == "sense.runtime.message.user.received"
    assert isinstance(first.payload, dict)
    assert first.payload.get("schema_version") == "1.0"
    assert second is None


def test_conscious_failure_does_not_emit_runtime_escalation(monkeypatch) -> None:
    bus = Bus()
    actions = ActionRegistry()
    actions.register("handle_conscious_message", lambda _ctx: _ExplodingAction())
    runtime = ActionExecutionRuntime(actions=actions, bus=bus, coordinator=_NoopCoordinator())

    monkeypatch.setattr(
        "alphonse.agent.actions.runtime._resolve_admin_telegram_target",
        lambda: "8553589429",
    )

    ctx = {
        "signal": Signal(
            type="sense.telegram.message.user.received",
            source="telegram",
            payload={"correlation_id": "corr-runtime-2"},
            correlation_id="corr-runtime-2",
        )
    }

    try:
        runtime.execute("handle_conscious_message", ctx)
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected runtime.execute to re-raise action failure")

    first = bus.get(timeout=0.1)
    assert first is None


def test_action_runtime_delivers_message_results(monkeypatch) -> None:
    bus = Bus()
    actions = ActionRegistry()
    actions.register("message", lambda _ctx: _MessageAction())
    adapter = _FakeAdapter()
    runtime = ActionExecutionRuntime(actions=actions, bus=bus, coordinator=_FakeCoordinator())

    monkeypatch.setattr(
        "alphonse.agent.actions.runtime.get_io_registry",
        lambda: _FakeRegistry(adapter),
    )

    result = runtime.execute("message", {"signal": Signal(type="test.message", source="system")})

    assert result is not None
    assert result.delivers_message is True
    assert len(adapter.deliveries) == 1
    assert adapter.deliveries[0].message == "hello"
