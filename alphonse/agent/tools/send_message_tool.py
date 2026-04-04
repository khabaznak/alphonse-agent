from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar
import uuid
from pathlib import Path

from alphonse.agent.cognition.plan_execution.communication_dispatcher import CommunicationDispatcher
from alphonse.agent.cognition.narration.outbound_narration_orchestrator import build_default_coordinator
from alphonse.agent.nervous_system import users as users_store
from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.services import communication_directory
from alphonse.agent.services.communication_service import CommunicationRequest, CommunicationService

logger = get_component_logger("tools.send_message_tool")


@dataclass(frozen=True)
class SendMessageTool:
    canonical_name: ClassVar[str] = "communication.send_message"
    capability: ClassVar[str] = "communication"
    _communication: CommunicationService | None = None

    def __post_init__(self) -> None:
        if self._communication is not None:
            return
        dispatcher = CommunicationDispatcher(coordinator=build_default_coordinator(), logger=logger)
        object.__setattr__(self, "_communication", CommunicationService(dispatcher=dispatcher))

    def execute(self, *, state: dict[str, Any] | None = None, **args: Any) -> dict[str, Any]:
        try:
            message = _get_message_from_args(args)
            recipient = _get_recipient_from_args(args)
            channel = _get_channel_from_args(args)
            urgency = _get_urgency_from_args(args)
            internal_progress = _get_internal_progress_from_args(args)
            visibility = _get_valid_visibility_from_args(args, internal_progress)
            outbound_intent = _get_outbound_intent_from_args(
                args,
                visibility=visibility,
                internal_progress=internal_progress,
            )
            delivery_mode = _get_delivery_mode_from_args(args)
            audio_file_path = _get_audio_file_path_from_args(args, delivery_mode=delivery_mode)
            as_voice = _get_as_voice_from_args(args)
            caption = _get_caption_from_args(args)
        except ValueError as exc:
            code = str(exc or "").strip()
            if code == "missing_message":
                return _failed(code="missing_message", message="message is required")
            if code == "missing_recipient":
                return _failed(code="missing_recipient", message="recipient is required")
            if code == "missing_audio_file_path":
                return _failed(code="missing_audio_file_path", message="audio file path is required for audio delivery mode")
            if code == "unresolved_recipient":
                return _failed(code="unresolved_recipient", message="recipient could not be resolved")
            raise

        state_payload = state if isinstance(state, dict) else {}
        origin_channel = str(state_payload.get("channel_type") or state_payload.get("channel") or "api").strip()
        origin_target = str(state_payload.get("channel_target") or state_payload.get("target") or "").strip() or None
        origin_service_id = communication_directory.resolve_service_id(origin_channel)
        explicit_service_id = communication_directory.resolve_service_id(channel)
        correlation_id = (
            str(state_payload.get("correlation_id") or "").strip() or str(args.get("correlation_id") or "").strip() or str(uuid.uuid4())
        )
        locale = str(state_payload.get("locale") or "").strip() or None

        request = CommunicationRequest(
            message=message,
            correlation_id=correlation_id,
            origin_channel=origin_channel,
            origin_target=origin_target,
            origin_service_id=origin_service_id,
            channel=channel,
            service_id=explicit_service_id,
            target=recipient.get("target"),
            user_id=recipient.get("user_id"),
            urgency=urgency,
            locale=locale,
        )
        exec_context = SimpleNamespace(
            channel_type=origin_channel,
            channel_target=origin_target,
            actor_person_id=str(recipient.get("user_id") or "").strip() or str(state_payload.get("actor_person_id") or "").strip() or None,
            correlation_id=correlation_id,
        )
        plan = SimpleNamespace(
            plan_id=f"tool-send-message:{correlation_id}",
            tool="communication.send_message",
            payload={
                **({"locale": locale} if locale else {}),
                **({"internal_progress": True} if internal_progress else {}),
                **({"visibility": visibility} if visibility else {}),
                **({"outbound_intent": outbound_intent} if outbound_intent else {}),
                **(
                    {
                        "delivery_mode": delivery_mode,
                        "audio_file_path": audio_file_path,
                        "as_voice": as_voice,
                        "caption": caption,
                    }
                    if delivery_mode == "audio"
                    else {}
                ),
            },
        )
        try:
            self._communication.send(
                request=request,
                context={"state": state_payload},
                exec_context=exec_context,
                plan=plan,
            )
        except ValueError as exc:
            return _map_send_error(exc)
        except Exception as exc:
            return _map_send_error(exc)

        return {
            "output": {
                "channel": channel or origin_channel,
                "recipient": recipient.get("target") or recipient.get("user_id"),
                "urgency": urgency,
                "visibility": visibility or "public",
                "outbound_intent": outbound_intent or "mission_public",
            },
            "exception": None,
            "metadata": {"tool": "communication.send_message"},
        }


@dataclass(frozen=True)
class SendVoiceNoteTool:
    canonical_name: ClassVar[str] = "communication.send_voice_note"
    capability: ClassVar[str] = "communication"
    _send_message_tool: SendMessageTool | None = None

    def __post_init__(self) -> None:
        if self._send_message_tool is not None:
            return
        object.__setattr__(self, "_send_message_tool", SendMessageTool())

    def execute(self, *, state: dict[str, Any] | None = None, **args: Any) -> dict[str, Any]:
        to = str(args.get("To") or args.get("to") or args.get("recipient") or "").strip()
        audio_file_path = str(args.get("AudioFilePath") or args.get("audio_file_path") or "").strip()
        caption = str(args.get("Caption") or args.get("caption") or "").strip()
        as_voice = bool(args.get("AsVoice") if args.get("AsVoice") is not None else args.get("as_voice", True))
        if as_voice and audio_file_path:
            suffix = Path(audio_file_path).suffix.lower()
            if suffix not in {".ogg", ".oga"}:
                return _failed(
                    code="voice_note_requires_ogg",
                    message="voice notes require .ogg or .oga audio file paths",
                )
        message = str(args.get("Message") or args.get("message") or "").strip()
        if not message:
            message = caption or "Voice note"
        return self._send_message_tool.execute(
            state=state,
            To=to,
            Message=message,
            Channel=str(args.get("Channel") or args.get("channel") or "").strip() or None,
            Urgency=str(args.get("Urgency") or args.get("urgency") or "normal").strip() or "normal",
            DeliveryMode="audio",
            AudioFilePath=audio_file_path,
            AsVoice=as_voice,
            Caption=caption or None,
            correlation_id=args.get("correlation_id"),
        )

def _get_message_from_args(args: dict[str, Any]) -> str:
    message = str(args.get("Message") or args.get("message") or "").strip()
    if not message:
        raise ValueError("missing_message")
    max_chars = _as_positive_int(args.get("MaxChars") if args.get("MaxChars") is not None else args.get("max_chars"))
    if max_chars > 0:
        message = _cap_message(message, limit=max_chars)
    if not message:
        raise ValueError("missing_message")
    return message


def _get_recipient_from_args(args: dict[str, Any]) -> dict[str, str | None]:
    rendered = str(args.get("To") or args.get("to") or args.get("recipient") or "").strip()
    if not rendered:
        raise ValueError("missing_recipient")
    if rendered.lstrip("-").isdigit():
        return {"user_id": None, "target": rendered}
    user = users_store.get_user(rendered)
    if isinstance(user, dict):
        user_id = str(user.get("user_id") or "").strip()
        if user_id:
            return {"user_id": user_id, "target": None}
    raise ValueError("unresolved_recipient")


def _get_channel_from_args(args: dict[str, Any]) -> str | None:
    return str(args.get("Channel") or args.get("channel") or "").strip().lower() or None


def _get_urgency_from_args(args: dict[str, Any]) -> str:
    return str(args.get("Urgency") or args.get("urgency") or "normal").strip().lower() or "normal"


def _get_internal_progress_from_args(args: dict[str, Any]) -> bool:
    return _as_bool(args.get("InternalProgress") if args.get("InternalProgress") is not None else args.get("internal_progress"))


def _get_valid_visibility_from_args(args: dict[str, Any], internal_progress: bool) -> str:
    visibility = str(args.get("Visibility") or args.get("visibility") or "").strip().lower()
    if internal_progress:
        return "internal"
    return visibility


def _get_outbound_intent_from_args(
    args: dict[str, Any],
    *,
    visibility: str,
    internal_progress: bool,
) -> str:
    outbound_intent = str(args.get("OutboundIntent") or args.get("outbound_intent") or "").strip().lower()
    if outbound_intent in {"wip_transition", "internal_progress", "mission_public"}:
        return outbound_intent
    return "internal_progress" if internal_progress or visibility == "internal" else "mission_public"


def _get_delivery_mode_from_args(args: dict[str, Any]) -> str:
    return str(args.get("DeliveryMode") or args.get("delivery_mode") or "text").strip().lower() or "text"


def _get_audio_file_path_from_args(args: dict[str, Any], *, delivery_mode: str) -> str | None:
    audio_file_path = str(args.get("AudioFilePath") or args.get("audio_file_path") or "").strip() or None
    if delivery_mode == "audio" and not audio_file_path:
        raise ValueError("missing_audio_file_path")
    return audio_file_path


def _get_as_voice_from_args(args: dict[str, Any]) -> bool:
    return bool(args.get("AsVoice") if args.get("AsVoice") is not None else args.get("as_voice", True))


def _get_caption_from_args(args: dict[str, Any]) -> str | None:
    return str(args.get("Caption") or args.get("caption") or "").strip() or None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    rendered = str(value or "").strip().lower()
    return rendered in {"1", "true", "yes", "on"}


def _as_positive_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _cap_message(message: str, *, limit: int) -> str:
    compact = str(message or "").strip()
    if len(compact) <= limit:
        return compact
    if limit <= 1:
        return compact[:limit]
    return compact[: limit - 1].rstrip() + "…"


def _failed(*, code: str, message: str) -> dict[str, Any]:
    return {
        "output": None,
        "exception": {
            "code": str(code),
            "message": str(message),
            "retryable": False,
            "details": {},
        },
        "metadata": {"tool": "communication.send_message"},
    }


def _map_send_error(exc: Exception) -> dict[str, Any]:
    rendered = str(exc or "").strip()
    code = rendered.lower() or "communication.send_message_failed"
    if code == "unresolved_recipient":
        return _failed(code="unresolved_recipient", message="recipient could not be resolved")
    if code == "missing_target":
        return _failed(code="missing_target", message="message target could not be resolved")
    if code == "missing_audio_file_path":
        return _failed(code="missing_audio_file_path", message="audio file path is required for audio delivery mode")
    if code.startswith("audio_file_not_found:"):
        missing_path = rendered.split(":", 1)[1] if ":" in rendered else ""
        message = f"audio file not found: {missing_path}" if missing_path else "audio file not found"
        return _failed(code="audio_file_not_found", message=message)
    return _failed(code="communication.send_message_failed", message=rendered or type(exc).__name__)
