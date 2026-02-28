from __future__ import annotations

from typing import Any

from alphonse.config import settings


class GetMySettingsTool:
    def execute(self, *, state: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        payload = dict(state or {})
        return {
            "status": "ok",
            "result": {
                "locale": str(payload.get("locale") or "").strip() or None,
                "tone": str(payload.get("tone") or "").strip() or None,
                "address_style": str(payload.get("address_style") or "").strip() or None,
                "timezone": str(payload.get("timezone") or settings.get_timezone() or "").strip() or None,
                "channel_type": str(payload.get("channel_type") or "").strip() or None,
                "execution_mode": str(settings.get_execution_mode() or "").strip() or None,
            },
            "error": None,
            "metadata": {"tool": "get_my_settings"},
        }


class GetUserDetailsTool:
    def execute(self, *, state: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        payload = dict(state or {})
        return {
            "status": "ok",
            "result": {
                "actor_person_id": str(payload.get("actor_person_id") or "").strip() or None,
                "incoming_user_id": str(payload.get("incoming_user_id") or "").strip() or None,
                "channel_type": str(payload.get("channel_type") or "").strip() or None,
                "channel_target": str(payload.get("channel_target") or "").strip() or None,
                "chat_id": str(payload.get("chat_id") or "").strip() or None,
                "conversation_key": str(payload.get("conversation_key") or "").strip() or None,
            },
            "error": None,
            "metadata": {"tool": "get_user_details"},
        }
