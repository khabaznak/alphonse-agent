from __future__ import annotations

from typing import Any

from alphonse.agent import identity
from alphonse.config import settings


class GetMySettingsTool:
    canonical_name: str = "get_my_settings"
    capability: str = "context"

    def execute(self, *, state: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        payload = dict(state or {})
        return {
            "output": {
                "locale": str(payload.get("locale") or "").strip() or None,
                "tone": str(payload.get("tone") or "").strip() or None,
                "address_style": str(payload.get("address_style") or "").strip() or None,
                "timezone": str(payload.get("timezone") or settings.get_timezone() or "").strip() or None,
                "channel_type": str(payload.get("channel_type") or "").strip() or None,
                "execution_mode": str(settings.get_execution_mode() or "").strip() or None,
            },
            "exception": None,
            "metadata": {"tool": "get_my_settings"},
        }


class GetUserDetailsTool:
    canonical_name: str = "get_user_details"
    capability: str = "context"

    def execute(self, *, state: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        payload = dict(state or {})
        canonical_user_id = identity.resolve_current_actor_user_id(payload)
        service_id = identity.resolve_service_id(
            str(payload.get("service_key") or payload.get("channel_type") or "").strip() or None
        )
        return {
            "output": {
                "actor_person_id": str(payload.get("actor_person_id") or "").strip() or None,
                "user_id": canonical_user_id,
                "resolved_user_id": canonical_user_id,
                "incoming_user_id": str(payload.get("incoming_user_id") or "").strip() or None,
                "channel_type": str(payload.get("channel_type") or "").strip() or None,
                "channel_target": str(payload.get("channel_target") or "").strip() or None,
                "service_id": service_id,
                "service_key": identity.resolve_service_key(service_id) or str(payload.get("channel_type") or "").strip() or None,
                "chat_id": str(payload.get("chat_id") or "").strip() or None,
                "conversation_key": str(payload.get("conversation_key") or "").strip() or None,
            },
            "exception": None,
            "metadata": {"tool": "get_user_details"},
        }
