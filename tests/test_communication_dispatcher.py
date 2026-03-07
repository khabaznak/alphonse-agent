from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from alphonse.agent.cognition.plan_execution.communication_dispatcher import CommunicationDispatcher
from alphonse.agent.io.contracts import NormalizedOutboundMessage


class _FakeCoordinator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def deliver(self, action: Any, context: dict[str, Any]) -> NormalizedOutboundMessage | None:
        payload = action.payload if isinstance(getattr(action, "payload", None), dict) else {}
        self.calls.append({"payload": dict(payload), "context": dict(context)})
        return NormalizedOutboundMessage(
            message=str(payload.get("message") or ""),
            channel_type=str(payload.get("channel_hint") or "telegram"),
            channel_target=str(payload.get("target") or "8553589429"),
            audience={"kind": "system", "id": "system"},
            correlation_id=str(payload.get("correlation_id") or ""),
            metadata={},
        )


def test_dispatch_step_message_suppresses_internal_progress_delivery(monkeypatch) -> None:
    coordinator = _FakeCoordinator()
    dispatcher = CommunicationDispatcher(coordinator=coordinator, logger=logging.getLogger("test.dispatcher"))
    delivered: list[NormalizedOutboundMessage] = []
    monkeypatch.setattr(dispatcher, "_deliver_normalized", lambda message: delivered.append(message))

    dispatcher.dispatch_step_message(
        channel="telegram",
        target="8553589429",
        message="Estoy avanzando",
        context={},
        exec_context=SimpleNamespace(channel_type="telegram", channel_target="8553589429", correlation_id="cid-1"),
        plan=SimpleNamespace(plan_id="p1", tool="send_message", payload={"internal_progress": True}),
    )

    assert coordinator.calls == []
    assert delivered == []


def test_dispatch_step_message_suppresses_internal_visibility_delivery(monkeypatch) -> None:
    coordinator = _FakeCoordinator()
    dispatcher = CommunicationDispatcher(coordinator=coordinator, logger=logging.getLogger("test.dispatcher"))
    delivered: list[NormalizedOutboundMessage] = []
    monkeypatch.setattr(dispatcher, "_deliver_normalized", lambda message: delivered.append(message))

    dispatcher.dispatch_step_message(
        channel="telegram",
        target="8553589429",
        message="nota interna",
        context={},
        exec_context=SimpleNamespace(channel_type="telegram", channel_target="8553589429", correlation_id="cid-2"),
        plan=SimpleNamespace(plan_id="p2", tool="send_message", payload={"visibility": "internal"}),
    )

    assert coordinator.calls == []
    assert delivered == []


def test_dispatch_step_message_delivers_public_mission(monkeypatch) -> None:
    coordinator = _FakeCoordinator()
    dispatcher = CommunicationDispatcher(coordinator=coordinator, logger=logging.getLogger("test.dispatcher"))
    delivered: list[NormalizedOutboundMessage] = []
    monkeypatch.setattr(dispatcher, "_deliver_normalized", lambda message: delivered.append(message))

    dispatcher.dispatch_step_message(
        channel="telegram",
        target="8553589429",
        message="Hola Alex",
        context={},
        exec_context=SimpleNamespace(channel_type="telegram", channel_target="8553589429", correlation_id="cid-3"),
        plan=SimpleNamespace(plan_id="p3", tool="send_message", payload={}),
    )

    assert len(coordinator.calls) == 1
    payload = coordinator.calls[0]["payload"]
    assert payload.get("outbound_intent") == "mission_public"
    assert payload.get("internal_progress") is False
    assert len(delivered) == 1
