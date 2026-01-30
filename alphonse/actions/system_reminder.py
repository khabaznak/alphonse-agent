from __future__ import annotations

from alphonse.actions.base import Action
from alphonse.actions.models import ActionResult


class SystemReminderAction(Action):
    key = "system_reminder"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        payload = getattr(signal, "payload", {}) if signal else {}
        message_payload = payload.get("payload") if isinstance(payload, dict) else {}
        message = "Timed signal triggered."
        if isinstance(message_payload, dict):
            message = str(message_payload.get("message") or message)

        target = None
        if isinstance(payload, dict):
            target = payload.get("target")

        return ActionResult(
            intention_key="NOTIFY_HOUSEHOLD",
            payload={
                "title": "Timed Signal",
                "message": message,
                "target_group": target or "all",
            },
            urgency="normal",
        )
