from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

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


def test_send_raises_when_recipient_ref_unresolved() -> None:
    dispatcher = _FakeDispatcher()
    service = CommunicationService(dispatcher=dispatcher)
    request = CommunicationRequest(
        message="hello",
        correlation_id="cid-1",
        origin_channel="telegram",
        origin_target="8553589429",
        channel="telegram",
        target=None,
        recipient_ref="this-name-does-not-exist",
    )
    with pytest.raises(ValueError, match="unresolved_recipient"):
        service.send(
            request=request,
            context={},
            exec_context=SimpleNamespace(channel_type="telegram", channel_target="8553589429", correlation_id="cid-1"),
            plan=SimpleNamespace(plan_id="p1", tool="sendMessage", payload={}),
        )
    assert dispatcher.called is False


def test_send_uses_origin_target_when_no_recipient_ref() -> None:
    dispatcher = _FakeDispatcher()
    service = CommunicationService(dispatcher=dispatcher)
    request = CommunicationRequest(
        message="hello",
        correlation_id="cid-2",
        origin_channel="telegram",
        origin_target="8553589429",
        channel="telegram",
        target=None,
        recipient_ref=None,
    )
    service.send(
        request=request,
        context={},
        exec_context=SimpleNamespace(channel_type="telegram", channel_target="8553589429", correlation_id="cid-2"),
        plan=SimpleNamespace(plan_id="p2", tool="sendMessage", payload={}),
    )
    assert dispatcher.called is True
    assert str((dispatcher.payload or {}).get("target") or "") == "8553589429"

