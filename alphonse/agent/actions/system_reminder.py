from __future__ import annotations

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult


class SystemReminderAction(Action):
    key = "system_reminder"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        payload = getattr(signal, "payload", {}) if signal else {}
        message_payload = payload.get("payload") if isinstance(payload, dict) else {}
        message = "Timed signal triggered."
        if isinstance(message_payload, dict):
            message = str(message_payload.get("message") or message)
        user_name = None
        if isinstance(message_payload, dict):
            user_name = message_payload.get("user_name")

        reminder_text = _format_reminder(message, user_name)

        origin = payload.get("origin") if isinstance(payload, dict) else None
        target = payload.get("target") if isinstance(payload, dict) else None
        correlation_id = payload.get("correlation_id") if isinstance(payload, dict) else None

        return ActionResult(
            intention_key="MESSAGE_READY",
            payload={
                "message": reminder_text,
                "origin": origin,
                "channel_hint": origin,
                "target": target,
                "audience": _audience_from_payload(payload),
                "correlation_id": correlation_id,
            },
            urgency="normal",
        )


def _format_reminder(message: str, user_name: str | None) -> str:
    if user_name:
        return f"I need to remind you {user_name} to {message}."
    return f"I need to remind you to {message}."


def _audience_from_payload(payload: dict) -> dict:
    person_id = payload.get("person_id") if isinstance(payload, dict) else None
    if person_id:
        return {"kind": "person", "id": person_id}
    return {"kind": "system", "id": "system"}
