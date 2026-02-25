from __future__ import annotations

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.nervous_system.pdca_queue_store import get_pdca_queue_metrics
from alphonse.agent.runtime import get_runtime


class HandleStatusAction(Action):
    key = "handle_status"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        correlation_id = getattr(signal, "correlation_id", None) if signal else None
        snapshot = get_runtime().snapshot()
        snapshot["pdca_metrics"] = get_pdca_queue_metrics()
        return ActionResult(
            intention_key="MESSAGE_READY",
            payload={
                "message": "ok",
                "origin": "api",
                "channel_hint": "api",
                "correlation_id": correlation_id,
                "data": snapshot,
                "audience": {"kind": "system", "id": "system"},
            },
            urgency="normal",
        )
