from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from alphonse.agent.nervous_system import user_service_resolvers as resolvers
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.io.contracts import NormalizedOutboundMessage
from alphonse.agent.services.communication_service import CommunicationRequest, CommunicationService


@dataclass
class _FakeCoordinator:
    calls: list[dict[str, Any]]

    def __init__(self) -> None:
        self.calls = []

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


def test_send_uses_origin_target_when_no_explicit_target_or_user() -> None:
    coordinator = _FakeCoordinator()
    service = CommunicationService(coordinator=coordinator)
    delivered: list[NormalizedOutboundMessage] = []
    service._deliver_normalized = lambda message: delivered.append(message)  # type: ignore[method-assign]
    request = CommunicationRequest(
        message="hello",
        correlation_id="cid-2",
        origin_channel="telegram",
        origin_target="8553589429",
        channel="telegram",
        target=None,
    )
    service.send(
        request=request,
        context={},
        exec_context=SimpleNamespace(channel_type="telegram", channel_target="8553589429", correlation_id="cid-2"),
        plan=SimpleNamespace(plan_id="p2", tool="communication.send_message", payload={}),
    )
    assert len(coordinator.calls) == 1
    assert str((coordinator.calls[0]["payload"].get("target") or "")) == "8553589429"
    assert len(delivered) == 1


def test_send_resolves_target_via_service_id_and_user_id(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    from alphonse.agent import identity as users_store

    users_store.upsert_user(
        {
            "user_id": "u-1",
            "principal_id": "p-1",
            "display_name": "Alex",
            "is_active": True,
        }
    )
    resolvers.upsert_service_resolver(
        user_id="u-1",
        service_id=2,
        service_user_id="8553589429",
        is_active=True,
    )

    coordinator = _FakeCoordinator()
    service = CommunicationService(coordinator=coordinator)
    delivered: list[NormalizedOutboundMessage] = []
    service._deliver_normalized = lambda message: delivered.append(message)  # type: ignore[method-assign]
    request = CommunicationRequest(
        message="hello",
        correlation_id="cid-3",
        origin_channel="telegram",
        origin_target="8553589429",
        service_id=2,
        user_id="u-1",
    )
    service.send(
        request=request,
        context={},
        exec_context=SimpleNamespace(channel_type="telegram", channel_target="8553589429", correlation_id="cid-3"),
        plan=SimpleNamespace(plan_id="p3", tool="communication.send_message", payload={}),
    )
    assert len(coordinator.calls) == 1
    payload = coordinator.calls[0]["payload"]
    assert str(payload.get("channel_hint") or "") == "telegram"
    assert str(payload.get("target") or "") == "8553589429"
    assert len(delivered) == 1


def test_send_suppresses_internal_progress_delivery() -> None:
    coordinator = _FakeCoordinator()
    service = CommunicationService(coordinator=coordinator)
    delivered: list[NormalizedOutboundMessage] = []
    service._deliver_normalized = lambda message: delivered.append(message)  # type: ignore[method-assign]
    request = CommunicationRequest(
        message="Estoy avanzando",
        correlation_id="cid-4",
        origin_channel="telegram",
        origin_target="8553589429",
        channel="telegram",
    )
    service.send(
        request=request,
        context={},
        exec_context=SimpleNamespace(channel_type="telegram", channel_target="8553589429", correlation_id="cid-4"),
        plan=SimpleNamespace(plan_id="p4", tool="communication.send_message", payload={"internal_progress": True}),
    )
    assert coordinator.calls == []
    assert delivered == []


def test_send_suppresses_internal_visibility_delivery() -> None:
    coordinator = _FakeCoordinator()
    service = CommunicationService(coordinator=coordinator)
    delivered: list[NormalizedOutboundMessage] = []
    service._deliver_normalized = lambda message: delivered.append(message)  # type: ignore[method-assign]
    request = CommunicationRequest(
        message="nota interna",
        correlation_id="cid-5",
        origin_channel="telegram",
        origin_target="8553589429",
        channel="telegram",
    )
    service.send(
        request=request,
        context={},
        exec_context=SimpleNamespace(channel_type="telegram", channel_target="8553589429", correlation_id="cid-5"),
        plan=SimpleNamespace(plan_id="p5", tool="communication.send_message", payload={"visibility": "internal"}),
    )
    assert coordinator.calls == []
    assert delivered == []


def test_send_delivers_public_mission_payload() -> None:
    coordinator = _FakeCoordinator()
    service = CommunicationService(coordinator=coordinator)
    delivered: list[NormalizedOutboundMessage] = []
    service._deliver_normalized = lambda message: delivered.append(message)  # type: ignore[method-assign]
    request = CommunicationRequest(
        message="Hola Alex",
        correlation_id="cid-6",
        origin_channel="telegram",
        origin_target="8553589429",
        channel="telegram",
    )
    service.send(
        request=request,
        context={},
        exec_context=SimpleNamespace(channel_type="telegram", channel_target="8553589429", correlation_id="cid-6"),
        plan=SimpleNamespace(plan_id="p6", tool="communication.send_message", payload={}),
    )
    assert len(coordinator.calls) == 1
    payload = coordinator.calls[0]["payload"]
    assert payload.get("outbound_intent") == "mission_public"
    assert payload.get("internal_progress") is False
    assert len(delivered) == 1


class _MissingAdapterRegistry:
    def get_extremity(self, channel_type: str):  # noqa: ANN201
        _ = channel_type
        return None


def test_send_raises_when_outbound_adapter_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    coordinator = _FakeCoordinator()
    service = CommunicationService(coordinator=coordinator)
    monkeypatch.setattr(
        "alphonse.agent.services.communication_service.get_io_registry",
        lambda: _MissingAdapterRegistry(),
    )
    request = CommunicationRequest(
        message="hello",
        correlation_id="cid-missing-adapter",
        origin_channel="api",
        origin_target="me",
        channel="api",
        target="me",
    )

    with pytest.raises(ValueError, match="missing_extremity_adapter:api"):
        service.send(
            request=request,
            context={},
            exec_context=SimpleNamespace(channel_type="api", channel_target="me", correlation_id="cid-missing-adapter"),
            plan=SimpleNamespace(plan_id="p-missing-adapter", tool="communication.send_message", payload={}),
        )
