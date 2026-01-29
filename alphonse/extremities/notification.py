from __future__ import annotations

from typing import Any

from core.integrations.fcm import send_push_notification
from core.integrations.webpush import send_web_push
from core.repositories.push_devices import list_active_push_devices
from core.settings_store import get_timezone
from alphonse.actions.models import ActionResult
from alphonse.extremities.base import Extremity


class NotificationExtremity(Extremity):
    def can_handle(self, result: ActionResult) -> bool:
        return result.intention_key in {"NOTIFY_HOUSEHOLD", "INFORM_STATUS"}

    def execute(self, result: ActionResult, narration: str | None = None) -> None:
        payload = result.payload
        title = payload.get("title") or "Alphonse"
        body = narration or payload.get("message") or ""
        target_group = payload.get("target_group") or "all"

        devices = list_active_push_devices(target_group, platforms=["android", "web"])
        android_tokens = []
        web_subscriptions: list[Any] = []
        for device in devices:
            platform = device.get("platform")
            token = device.get("token")
            if platform == "android" and isinstance(token, str):
                android_tokens.append(token)
            elif platform == "web" and token:
                web_subscriptions.append(token)

        data = {
            "intention": result.intention_key,
            "timezone": get_timezone(),
        }

        if android_tokens:
            send_push_notification(android_tokens, title, body, data=data)

        for subscription in web_subscriptions:
            send_web_push(_normalize_subscription(subscription), title, body, data=data)


def _normalize_subscription(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        import json

        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid web push subscription JSON.") from exc
    raise ValueError("Unsupported web push subscription format.")
