from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphonse.agent.cognition.plan_execution.communication_dispatcher import CommunicationDispatcher
from alphonse.agent.cognition.preferences.store import get_preference, get_principal_for_channel
from alphonse.agent.nervous_system import user_service_resolvers as service_resolvers
from alphonse.agent.nervous_system.services import TELEGRAM_SERVICE_ID
from alphonse.agent.nervous_system import users as users_store


@dataclass(frozen=True)
class CommunicationRequest:
    message: str
    correlation_id: str
    origin_channel: str
    origin_target: str | None
    channel: str | None = None
    target: str | None = None
    recipient_ref: str | None = None
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
        channel = self._resolve_channel(request)
        target = self._resolve_target(request, channel=channel)
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

    def _resolve_channel(self, request: CommunicationRequest) -> str:
        explicit = str(request.channel or "").strip().lower()
        if explicit:
            return explicit
        principal_id = get_principal_for_channel(
            str(request.origin_channel or "").strip(),
            str(request.origin_target or "").strip(),
        )
        preferred = ""
        if principal_id:
            value = get_preference(principal_id, "preferred_communication_channel")
            preferred = str(value or "").strip().lower()
        if preferred:
            return preferred
        return str(request.origin_channel or "telegram").strip().lower() or "telegram"

    def _resolve_target(self, request: CommunicationRequest, *, channel: str) -> str | None:
        explicit_target = str(request.target or "").strip()
        if explicit_target:
            return explicit_target
        ref = str(request.recipient_ref or "").strip()
        if not ref:
            return str(request.origin_target or "").strip() or None
        if ref.lstrip("-").isdigit():
            return ref
        if channel == "telegram":
            resolved = service_resolvers.resolve_telegram_chat_id_for_user(ref)
            if resolved:
                return resolved
        user = users_store.get_user_by_display_name(ref)
        if isinstance(user, dict):
            user_id = str(user.get("user_id") or "").strip()
            if channel == "telegram" and user_id:
                resolved = service_resolvers.resolve_service_user_id(
                    user_id=user_id,
                    service_id=TELEGRAM_SERVICE_ID,
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
