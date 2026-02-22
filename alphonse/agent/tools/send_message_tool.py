from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
import uuid
import logging

from alphonse.agent.cognition.plan_execution.communication_dispatcher import CommunicationDispatcher
from alphonse.agent.cognition.narration.outbound_narration_orchestrator import build_default_coordinator
from alphonse.agent.services.communication_service import CommunicationRequest, CommunicationService


@dataclass(frozen=True)
class SendMessageTool:
    _communication: CommunicationService | None = None

    def __post_init__(self) -> None:
        if self._communication is not None:
            return
        dispatcher = CommunicationDispatcher(coordinator=build_default_coordinator(), logger=logging.getLogger(__name__))
        object.__setattr__(self, "_communication", CommunicationService(dispatcher=dispatcher))

    def execute(self, *, state: dict[str, Any] | None = None, **args: Any) -> dict[str, Any]:
        message = str(args.get("Message") or args.get("message") or "").strip()
        to = str(args.get("To") or args.get("to") or args.get("recipient") or "").strip()
        channel = str(args.get("Channel") or args.get("channel") or "").strip().lower() or None
        urgency = str(args.get("Urgency") or args.get("urgency") or "normal").strip().lower() or "normal"
        if not message:
            return _failed(code="missing_message", message="message is required")
        if not to:
            return _failed(code="missing_recipient", message="recipient is required")

        state_payload = state if isinstance(state, dict) else {}
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
            payload={"locale": locale} if locale else {},
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
