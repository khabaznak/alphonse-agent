from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphonse.agent.tools.registry import build_default_tool_registry
from alphonse.agent.tools.registry2 import build_planner_tool_registry
from alphonse.agent.tools.send_message_tool import SendMessageTool


@dataclass
class _FakeCommunication:
    called: bool = False
    request: Any = None

    def send(self, *, request: Any, context: dict[str, Any], exec_context: Any, plan: Any) -> None:
        _ = (context, exec_context, plan)
        self.called = True
        self.request = request


def test_send_message_exposed_in_registry2() -> None:
    registry = build_planner_tool_registry()
    assert registry.get("sendMessage") is not None


def test_send_message_registered_in_runtime_registry() -> None:
    registry = build_default_tool_registry()
    assert registry.get("sendMessage") is not None


def test_send_message_tool_executes_delivery() -> None:
    fake = _FakeCommunication()
    tool = SendMessageTool(_communication=fake)
    result = tool.execute(
        state={"channel_type": "telegram", "channel_target": "8553589429", "correlation_id": "cid-1"},
        To="Gabriela",
        Message="Hola Gaby",
        Channel="telegram",
    )
    assert result["status"] == "ok"
    assert fake.called is True
    assert str(fake.request.channel) == "telegram"
    assert str(fake.request.recipient_ref) == "Gabriela"


def test_send_message_tool_validates_required_fields() -> None:
    tool = SendMessageTool(_communication=_FakeCommunication())
    result = tool.execute(state={"channel_type": "telegram"}, To="", Message="")
    assert result["status"] == "failed"
    assert str((result.get("error") or {}).get("code") or "") in {"missing_message", "missing_recipient"}


@dataclass
class _FailingCommunication:
    code: str

    def send(self, *, request: Any, context: dict[str, Any], exec_context: Any, plan: Any) -> None:
        _ = (request, context, exec_context, plan)
        raise ValueError(self.code)


def test_send_message_tool_maps_unresolved_recipient_error() -> None:
    tool = SendMessageTool(_communication=_FailingCommunication(code="unresolved_recipient"))
    result = tool.execute(
        state={"channel_type": "telegram", "channel_target": "8553589429"},
        To="Gabriela",
        Message="Hola Gaby",
        Channel="telegram",
    )
    assert result["status"] == "failed"
    assert str((result.get("error") or {}).get("code") or "") == "unresolved_recipient"


def test_send_message_tool_maps_first_contact_from_user_search() -> None:
    fake = _FakeCommunication()
    tool = SendMessageTool(_communication=fake)
    state = {
        "channel_type": "telegram",
        "channel_target": "8553589429",
        "task_state": {
            "facts": {
                "step_1": {
                    "tool": "user_search",
                    "result": {
                        "status": "ok",
                        "result": {
                            "users": [
                                {
                                    "user_id": "u-1",
                                    "display_name": "Gabriela Perez",
                                    "telegram_user_id": "999111222",
                                }
                            ]
                        },
                    },
                }
            }
        },
    }
    result = tool.execute(
        state=state,
        To="the first contact",
        Message="Hola Gaby",
        Channel="telegram",
    )
    assert result["status"] == "ok"
    assert fake.called is True
    assert str(fake.request.target) == "999111222"
    assert str(fake.request.recipient_ref or "") == ""


def test_send_message_tool_maps_partial_name_from_user_search() -> None:
    fake = _FakeCommunication()
    tool = SendMessageTool(_communication=fake)
    state = {
        "channel_type": "telegram",
        "channel_target": "8553589429",
        "task_state": {
            "facts": {
                "step_1": {
                    "tool": "user_search",
                    "result": {
                        "status": "ok",
                        "result": {
                            "users": [
                                {
                                    "user_id": "u-1",
                                    "display_name": "Gabriela Perez",
                                    "telegram_user_id": "999111222",
                                }
                            ]
                        },
                    },
                }
            }
        },
    }
    result = tool.execute(
        state=state,
        To="Gabriela",
        Message="Hola Gaby",
        Channel="telegram",
    )
    assert result["status"] == "ok"
    assert fake.called is True
    assert str(fake.request.target) == "999111222"
    assert str(fake.request.recipient_ref or "") == ""
