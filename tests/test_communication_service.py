from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from alphonse.agent.nervous_system import user_service_resolvers as resolvers
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.services.communication_service import CommunicationRequest, CommunicationService


@dataclass
class _FakeDispatcher:
    called: bool = False
    payload: dict[str, Any] | None = None

    def dispatch_step_message(
        self,
        *,
        channel: str,
        target: str | None,
        message: str,
        context: dict[str, Any],
        exec_context: Any,
        plan: Any,
    ) -> None:
        _ = (context, exec_context, plan)
        self.called = True
        self.payload = {"channel": channel, "target": target, "message": message}


def test_send_uses_origin_target_when_no_explicit_target_or_user() -> None:
    dispatcher = _FakeDispatcher()
    service = CommunicationService(dispatcher=dispatcher)
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
    assert dispatcher.called is True
    assert str((dispatcher.payload or {}).get("target") or "") == "8553589429"


def test_send_resolves_target_via_service_id_and_user_id(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    from alphonse.agent.nervous_system import users as users_store

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

    dispatcher = _FakeDispatcher()
    service = CommunicationService(dispatcher=dispatcher)
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
    assert dispatcher.called is True
    assert str((dispatcher.payload or {}).get("channel") or "") == "telegram"
    assert str((dispatcher.payload or {}).get("target") or "") == "8553589429"
