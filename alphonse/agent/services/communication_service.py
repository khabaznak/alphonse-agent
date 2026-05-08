from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphonse.agent import identity
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.cognition.narration.outbound_narration_orchestrator import DeliveryCoordinator
from alphonse.agent.cognition.preferences.store import get_user_preference
from alphonse.agent.io import NormalizedOutboundMessage, get_io_registry
from alphonse.agent.observability.log_manager import get_log_manager

_LOG = get_log_manager()


@dataclass(frozen=True)
class CommunicationRequest:
    message: str
    correlation_id: str
    origin_channel: str | None
    origin_target: str | None
    origin_service_id: int | None = None
    channel: str | None = None
    service_id: int | None = None
    target: str | None = None
    user_id: str | None = None
    urgency: str = "normal"
    locale: str | None = None


class CommunicationService:
    def __init__(self, *, coordinator: DeliveryCoordinator) -> None:
        self._coordinator = coordinator

    def send(
        self,
        *,
        request: CommunicationRequest,
        context: dict[str, Any],
        exec_context: Any,
        plan: Any,
    ) -> None:
        resolved_service_id = self._resolve_service_id(request)
        channel = self._resolve_channel(request, service_id=resolved_service_id)
        target = self._resolve_target(request, service_id=resolved_service_id)
        if not str(target or "").strip():
            raise ValueError("missing_target")
        if self._is_blocked_by_policy(request=request, target=target):
            return
        message = self._apply_tone(request=request, target=target)
        self._dispatch_step_message(
            channel=channel,
            target=target,
            message=message,
            context=context,
            exec_context=exec_context,
            plan=plan,
        )

    def _dispatch_step_message(
        self,
        *,
        channel: str,
        target: str | None,
        message: str,
        context: dict[str, Any],
        exec_context: Any,
        plan: Any,
    ) -> None:
        plan_id = str(getattr(plan, "plan_id", "") or "")
        step = str(getattr(plan, "tool", "") or "unknown")
        payload_dict = getattr(plan, "payload", {}) if hasattr(plan, "payload") else {}
        outbound_intent = _derive_outbound_intent(payload_dict)
        internal_progress = _as_bool(payload_dict.get("internal_progress")) if isinstance(payload_dict, dict) else False
        visibility = str(payload_dict.get("visibility") or "").strip().lower() if isinstance(payload_dict, dict) else ""
        if not target:
            return
        if outbound_intent == "internal_progress":
            return
        payload = _message_payload(message, channel, target, exec_context)
        payload["outbound_intent"] = outbound_intent
        payload["internal_progress"] = internal_progress
        if visibility:
            payload["visibility"] = visibility
        if isinstance(payload_dict, dict) and payload_dict.get("locale"):
            payload["locale"] = payload_dict.get("locale")
        if isinstance(payload_dict, dict):
            for key in ("delivery_mode", "audio_file_path", "as_voice", "caption"):
                if key in payload_dict:
                    payload[key] = payload_dict.get(key)
        action = ActionResult(
            intention_key="MESSAGE_READY",
            payload=payload,
            urgency="normal",
            delivers_message=True,
        )
        delivery = self._coordinator.deliver(action, context)
        if delivery:
            self._deliver_normalized(delivery)

    def _deliver_normalized(self, delivery: NormalizedOutboundMessage) -> None:
        registry = get_io_registry()
        adapter = registry.get_extremity(delivery.channel_type)
        if not adapter:
            channel = str(delivery.channel_type or "").strip() or "unknown"
            _LOG.emit(
                level="error",
                event="communication.delivery.failed",
                component="services.communication_service",
                correlation_id=str(delivery.correlation_id or "").strip() or None,
                channel=channel or None,
                payload={
                    "reason": "missing_extremity_adapter",
                    "channel_target": str(delivery.channel_target or "").strip() or None,
                },
            )
            raise ValueError(f"missing_extremity_adapter:{channel}")
        adapter.deliver(delivery)

    def _resolve_service_id(self, request: CommunicationRequest) -> int | None:
        explicit_service_id = request.service_id
        if explicit_service_id is not None:
            return int(explicit_service_id)
        explicit_channel = str(request.channel or "").strip().lower()
        if explicit_channel:
            resolved = identity.resolve_service_id(explicit_channel)
            if resolved is not None:
                return resolved
        user_id = str(request.user_id or "").strip()
        if user_id:
            preferred = identity.get_preferred_service_id(user_id)
            if preferred is not None:
                return preferred
        origin_service_id = request.origin_service_id
        if origin_service_id is not None:
            return int(origin_service_id)
        origin_channel = str(request.origin_channel or "").strip().lower()
        if origin_channel:
            return identity.resolve_service_id(origin_channel)
        return None

    def _resolve_channel(self, request: CommunicationRequest, *, service_id: int | None) -> str:
        resolved_key = identity.resolve_service_key(service_id)
        if resolved_key:
            return resolved_key
        explicit = str(request.channel or "").strip().lower()
        if explicit:
            return explicit
        return str(request.origin_channel or "").strip().lower()

    def _resolve_target(self, request: CommunicationRequest, *, service_id: int | None) -> str | None:
        explicit_target = str(request.target or "").strip()
        if explicit_target:
            return explicit_target
        user_id = str(request.user_id or "").strip()
        if user_id and service_id is not None:
            resolved = identity.resolve_delivery_target(
                user_id=user_id,
                service_id=service_id,
            )
            if resolved:
                return resolved
        return str(request.origin_target or "").strip() or None

    def _is_blocked_by_policy(self, *, request: CommunicationRequest, target: str | None) -> bool:
        if str(request.urgency or "").strip().lower() in {"urgent", "critical"}:
            return False
        user_id = str(request.user_id or "").strip()
        if not user_id:
            return False
        comm_mode = str(get_user_preference(user_id, "communication_mode") or "").strip().lower()
        return comm_mode in {"dnd", "sleep"}

    def _apply_tone(self, *, request: CommunicationRequest, target: str | None) -> str:
        message = str(request.message or "").strip()
        if not message:
            return ""
        user_id = str(request.user_id or "").strip()
        if not user_id:
            return message
        tone = str(get_user_preference(user_id, "tone") or "").strip().lower()
        if tone == "formal":
            return message
        return message


def _message_payload(
    message: str,
    channel: str,
    target: str | None,
    exec_context: Any,
) -> dict[str, Any]:
    actor_person_id = str(getattr(exec_context, "actor_person_id", "") or "").strip()
    audience = {"kind": "person", "id": actor_person_id} if actor_person_id else {"kind": "system", "id": "system"}
    payload: dict[str, Any] = {
        "message": message,
        "origin": channel,
        "channel_hint": channel,
        "correlation_id": str(getattr(exec_context, "correlation_id", "") or ""),
        "audience": audience,
    }
    if target:
        payload["target"] = target
    return payload


def _derive_outbound_intent(payload: dict[str, Any] | Any) -> str:
    if not isinstance(payload, dict):
        return "mission_public"
    configured = str(payload.get("outbound_intent") or "").strip().lower()
    if configured in {"wip_transition", "internal_progress", "mission_public"}:
        return configured
    if _as_bool(payload.get("internal_progress")):
        return "internal_progress"
    if str(payload.get("visibility") or "").strip().lower() == "internal":
        return "internal_progress"
    return "mission_public"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
