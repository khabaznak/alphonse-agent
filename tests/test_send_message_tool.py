from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphonse.agent.tools.registry import build_default_tool_registry
from alphonse.agent.tools.registry2 import build_planner_tool_registry
from alphonse.agent.tools.send_message_tool import SendMessageTool
from alphonse.agent.tools.send_message_tool import SendVoiceNoteTool


@dataclass
class _FakeCommunication:
    called: bool = False
    request: Any = None
    plan: Any = None

    def send(self, *, request: Any, context: dict[str, Any], exec_context: Any, plan: Any) -> None:
        _ = (context, exec_context, plan)
        self.called = True
        self.request = request
        self.plan = plan


def test_send_message_exposed_in_registry2() -> None:
    registry = build_planner_tool_registry()
    assert registry.get("send_message") is not None


def test_send_message_registered_in_runtime_registry() -> None:
    registry = build_default_tool_registry()
    assert registry.get("sendMessage") is not None


def test_send_voice_note_exposed_in_registry2() -> None:
    registry = build_planner_tool_registry()
    assert registry.get("send_voice_note") is not None


def test_send_voice_note_registered_in_runtime_registry() -> None:
    registry = build_default_tool_registry()
    assert registry.get("sendVoiceNote") is not None


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
    assert str(fake.request.recipient_ref) == "u-1"
    assert str(fake.request.target or "") == ""


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
    assert str(fake.request.recipient_ref) == "u-1"
    assert str(fake.request.target or "") == ""


def test_send_message_audio_requires_audio_file_path() -> None:
    tool = SendMessageTool(_communication=_FakeCommunication())
    result = tool.execute(
        state={"channel_type": "telegram", "channel_target": "8553589429"},
        To="Gabriela",
        Message="Hola Gaby",
        DeliveryMode="audio",
    )
    assert result["status"] == "failed"
    assert str((result.get("error") or {}).get("code") or "") == "missing_audio_file_path"


def test_send_message_audio_payload_is_added_to_plan() -> None:
    fake = _FakeCommunication()
    tool = SendMessageTool(_communication=fake)
    result = tool.execute(
        state={"channel_type": "telegram", "channel_target": "8553589429", "correlation_id": "cid-aud"},
        To="Gabriela",
        Message="Hola Gaby",
        Channel="telegram",
        DeliveryMode="audio",
        AudioFilePath="/tmp/alphonse-audio/response-1.m4a",
        AsVoice=False,
        Caption="Hola por audio",
    )
    assert result["status"] == "ok"
    assert fake.called is True
    payload = dict(getattr(fake.plan, "payload", {}) or {})
    assert payload.get("delivery_mode") == "audio"
    assert payload.get("audio_file_path") == "/tmp/alphonse-audio/response-1.m4a"
    assert payload.get("as_voice") is False
    assert payload.get("caption") == "Hola por audio"


def test_send_message_tool_maps_audio_file_not_found_error() -> None:
    tool = SendMessageTool(_communication=_FailingCommunication(code="audio_file_not_found:/tmp/missing.m4a"))
    result = tool.execute(
        state={"channel_type": "telegram", "channel_target": "8553589429"},
        To="Gabriela",
        Message="Hola Gaby",
        Channel="telegram",
    )
    assert result["status"] == "failed"
    assert str((result.get("error") or {}).get("code") or "") == "audio_file_not_found"


def test_send_voice_note_tool_enforces_audio_delivery_mode() -> None:
    fake = _FakeCommunication()
    tool = SendVoiceNoteTool(_send_message_tool=SendMessageTool(_communication=fake))
    result = tool.execute(
        state={"channel_type": "telegram", "channel_target": "8553589429", "correlation_id": "cid-voice"},
        To="Gabriela",
        AudioFilePath="/tmp/alphonse-audio/voice-1.ogg",
        Caption="Hola por voz",
        Channel="telegram",
    )
    assert result["status"] == "ok"
    assert fake.called is True
    payload = dict(getattr(fake.plan, "payload", {}) or {})
    assert payload.get("delivery_mode") == "audio"
    assert payload.get("audio_file_path") == "/tmp/alphonse-audio/voice-1.ogg"
    assert payload.get("as_voice") is True


def test_send_voice_note_tool_rejects_non_ogg_for_voice_notes() -> None:
    fake = _FakeCommunication()
    tool = SendVoiceNoteTool(_send_message_tool=SendMessageTool(_communication=fake))
    result = tool.execute(
        state={"channel_type": "telegram", "channel_target": "8553589429", "correlation_id": "cid-voice-bad"},
        To="Gabriela",
        AudioFilePath="/tmp/alphonse-audio/voice-1.m4a",
        Channel="telegram",
        AsVoice=True,
    )
    assert result["status"] == "failed"
    assert str((result.get("error") or {}).get("code") or "") == "voice_note_requires_ogg"
    assert fake.called is False
