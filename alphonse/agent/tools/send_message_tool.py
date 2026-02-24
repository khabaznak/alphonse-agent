from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
import uuid
import re

from alphonse.agent.cognition.plan_execution.communication_dispatcher import CommunicationDispatcher
from alphonse.agent.cognition.narration.outbound_narration_orchestrator import build_default_coordinator
from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.services.communication_service import CommunicationRequest, CommunicationService

logger = get_component_logger("tools.send_message_tool")

@dataclass(frozen=True)
class SendMessageTool:
    _communication: CommunicationService | None = None

    def __post_init__(self) -> None:
        if self._communication is not None:
            return
        dispatcher = CommunicationDispatcher(coordinator=build_default_coordinator(), logger=logger)
        object.__setattr__(self, "_communication", CommunicationService(dispatcher=dispatcher))

    def execute(self, *, state: dict[str, Any] | None = None, **args: Any) -> dict[str, Any]:
        message = str(args.get("Message") or args.get("message") or "").strip()
        to = str(args.get("To") or args.get("to") or args.get("recipient") or "").strip()
        channel = str(args.get("Channel") or args.get("channel") or "").strip().lower() or None
        urgency = str(args.get("Urgency") or args.get("urgency") or "normal").strip().lower() or "normal"
        delivery_mode = str(args.get("DeliveryMode") or args.get("delivery_mode") or "text").strip().lower() or "text"
        audio_file_path = str(args.get("AudioFilePath") or args.get("audio_file_path") or "").strip() or None
        as_voice = bool(args.get("AsVoice") if args.get("AsVoice") is not None else args.get("as_voice", True))
        caption = str(args.get("Caption") or args.get("caption") or "").strip() or None
        if not message:
            return _failed(code="missing_message", message="message is required")
        if not to:
            return _failed(code="missing_recipient", message="recipient is required")
        if delivery_mode == "audio" and not audio_file_path:
            return _failed(code="missing_audio_file_path", message="audio file path is required for audio delivery mode")

        state_payload = state if isinstance(state, dict) else {}
        to = _resolve_recipient_ref(to=to, state=state_payload)
        origin_channel = str(state_payload.get("channel_type") or state_payload.get("channel") or "api").strip()
        origin_target = str(state_payload.get("channel_target") or state_payload.get("target") or "").strip() or None
        correlation_id = (
            str(state_payload.get("correlation_id") or "").strip() or str(args.get("correlation_id") or "").strip() or str(uuid.uuid4())
        )
        locale = str(state_payload.get("locale") or "").strip() or None

        request = CommunicationRequest(
            message=message,
            correlation_id=correlation_id,
            origin_channel=origin_channel,
            origin_target=origin_target,
            channel=channel,
            target=to if to.lstrip("-").isdigit() else None,
            recipient_ref=to if not to.lstrip("-").isdigit() else None,
            urgency=urgency,
            locale=locale,
        )
        exec_context = SimpleNamespace(
            channel_type=origin_channel,
            channel_target=origin_target,
            actor_person_id=None,
            correlation_id=correlation_id,
        )
        plan = SimpleNamespace(
            plan_id=f"tool-send-message:{correlation_id}",
            tool="sendMessage",
            payload={
                **({"locale": locale} if locale else {}),
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
            code = str(exc or "").strip().lower() or "send_message_failed"
            if code == "unresolved_recipient":
                return _failed(code="unresolved_recipient", message="recipient could not be resolved")
            if code == "missing_target":
                return _failed(code="missing_target", message="message target could not be resolved")
            return _failed(code="send_message_failed", message=str(exc) or type(exc).__name__)
        except Exception as exc:
            return _failed(code="send_message_failed", message=str(exc) or type(exc).__name__)

        return {
            "status": "ok",
            "result": {
                "channel": channel or origin_channel,
                "recipient": to,
                "urgency": urgency,
            },
            "error": None,
            "metadata": {"tool": "sendMessage"},
        }


def _resolve_recipient_ref(*, to: str, state: dict[str, Any]) -> str:
    rendered = str(to or "").strip()
    if not rendered:
        return ""
    if rendered.lstrip("-").isdigit():
        return rendered
    search_rows = _latest_user_search_rows(state)
    if not search_rows:
        return rendered
    by_index = _resolve_by_ordinal_reference(rendered, search_rows)
    if by_index:
        return by_index
    by_name = _resolve_by_display_name(rendered, search_rows)
    if by_name:
        return by_name
    return rendered


def _latest_user_search_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    task_state = state.get("task_state")
    if not isinstance(task_state, dict):
        return []
    facts = task_state.get("facts")
    if not isinstance(facts, dict):
        return []
    for _, entry in reversed(list(facts.items())):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("tool") or "").strip() != "user_search":
            continue
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        payload = result.get("result")
        if not isinstance(payload, dict):
            continue
        users = payload.get("users")
        if isinstance(users, list):
            return [row for row in users if isinstance(row, dict)]
    return []


def _resolve_by_ordinal_reference(to: str, users: list[dict[str, Any]]) -> str | None:
    normalized = _normalize_text(to)
    patterns = [
        (r"\bfirst\b|\b1st\b|\bone\b", 0),
        (r"\bsecond\b|\b2nd\b|\btwo\b", 1),
        (r"\bthird\b|\b3rd\b|\bthree\b", 2),
    ]
    for pattern, index in patterns:
        if re.search(pattern, normalized):
            if index < len(users):
                return _user_delivery_ref(users[index])
            return None
    return None


def _resolve_by_display_name(to: str, users: list[dict[str, Any]]) -> str | None:
    needle = _normalize_text(to)
    if not needle:
        return None
    matches: list[dict[str, Any]] = []
    for user in users:
        name = _normalize_text(str(user.get("display_name") or ""))
        if not name:
            continue
        if needle in name or name in needle:
            matches.append(user)
    if len(matches) == 1:
        return _user_delivery_ref(matches[0])
    return None


def _user_delivery_ref(user: dict[str, Any]) -> str:
    telegram_user_id = str(user.get("telegram_user_id") or "").strip()
    if telegram_user_id:
        return telegram_user_id
    user_id = str(user.get("user_id") or "").strip()
    if user_id:
        return user_id
    return str(user.get("display_name") or "").strip()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _failed(*, code: str, message: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "result": None,
        "error": {
            "code": str(code),
            "message": str(message),
            "retryable": False,
            "details": {},
        },
        "metadata": {"tool": "sendMessage"},
    }
