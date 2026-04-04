from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphonse.agent.cognition.plan_execution.communication_dispatcher import CommunicationDispatcher
from alphonse.agent.cognition.preferences.store import get_preference, get_principal_for_channel
from alphonse.agent.nervous_system import users as users_store
from alphonse.agent.services import communication_directory


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
    def __init__(self, *, dispatcher: CommunicationDispatcher) -> None:
        self._dispatcher = dispatcher

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
        self._dispatcher.dispatch_step_message(
            channel=channel,
            target=target,
            message=message,
            context=context,
            exec_context=exec_context,
            plan=plan,
        )

    def _resolve_service_id(self, request: CommunicationRequest) -> int | None:
        explicit_service_id = request.service_id
        if explicit_service_id is not None:
            return int(explicit_service_id)
        explicit_channel = str(request.channel or "").strip().lower()
        if explicit_channel:
            resolved = communication_directory.resolve_service_id(explicit_channel)
            if resolved is not None:
                return resolved
        user_id = str(request.user_id or "").strip()
        if user_id:
            preferred = communication_directory.get_preferred_service_id(user_id)
            if preferred is not None:
                return preferred
        origin_service_id = request.origin_service_id
        if origin_service_id is not None:
            return int(origin_service_id)
        principal_id = get_principal_for_channel(
            str(request.origin_channel or "").strip(),
            str(request.origin_target or "").strip(),
        )
        if principal_id:
            value = get_preference(principal_id, "preferred_communication_channel")
            preferred = communication_directory.resolve_service_id(str(value or "").strip())
            if preferred is not None:
                return preferred
        origin_channel = str(request.origin_channel or "").strip().lower()
        if origin_channel:
            return communication_directory.resolve_service_id(origin_channel)
        return None

    def _resolve_channel(self, request: CommunicationRequest, *, service_id: int | None) -> str:
        resolved_key = communication_directory.resolve_service_key(service_id)
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
            resolved = communication_directory.resolve_delivery_target(
                user_id=user_id,
                service_id=service_id,
            )
            if resolved:
                return resolved
        return str(request.origin_target or "").strip() or None

    def _is_blocked_by_policy(self, *, request: CommunicationRequest, target: str | None) -> bool:
        if str(request.urgency or "").strip().lower() in {"urgent", "critical"}:
            return False
        principal_id = get_principal_for_channel(
            str(request.origin_channel or "").strip(),
            str(target or request.origin_target or "").strip(),
        )
        if not principal_id:
            return False
        comm_mode = str(get_preference(principal_id, "communication_mode") or "").strip().lower()
        return comm_mode in {"dnd", "sleep"}

    def _apply_tone(self, *, request: CommunicationRequest, target: str | None) -> str:
        message = str(request.message or "").strip()
        if not message:
            return ""
        principal_id = get_principal_for_channel(
            str(request.origin_channel or "").strip(),
            str(target or request.origin_target or "").strip(),
        )
        if not principal_id:
            return message
        tone = str(get_preference(principal_id, "tone") or "").strip().lower()
        if tone == "formal":
            return message
        return message
