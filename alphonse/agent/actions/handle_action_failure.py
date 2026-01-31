from __future__ import annotations

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult


class HandleActionFailure(Action):
    key = "handle_action_failure"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        payload = getattr(signal, "payload", {}) if signal else {}
        error_message = payload.get("error_message") or "Action failed"
        correlation_id = getattr(signal, "correlation_id", None) if signal else None
        return ActionResult(
            intention_key="MESSAGE_READY",
            payload={
                "message": str(error_message),
                "origin": "system",
                "channel_hint": "silent",
                "correlation_id": correlation_id,
                "audience": {"kind": "system", "id": "system"},
            },
            urgency="normal",
        )
