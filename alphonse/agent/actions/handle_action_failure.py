from __future__ import annotations

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult


class HandleActionFailure(Action):
    key = "handle_action_failure"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        payload = getattr(signal, "payload", {}) if signal else {}
        error_message = payload.get("error_message") or "Action failed"
        return ActionResult(
            intention_key="NOTIFY_HOUSEHOLD",
            payload={
                "title": "Action Failed",
                "message": str(error_message),
                "target_group": "all",
            },
            urgency="normal",
        )
