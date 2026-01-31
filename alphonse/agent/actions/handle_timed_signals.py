from __future__ import annotations

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.nervous_system.timed_store import list_timed_signals


class HandleTimedSignalsAction(Action):
    key = "handle_timed_signals"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        payload = getattr(signal, "payload", {}) if signal else {}
        correlation_id = getattr(signal, "correlation_id", None) if signal else None
        limit = payload.get("limit") or 200
        try:
            limit_val = int(limit)
        except (TypeError, ValueError):
            limit_val = 200
        data = {"timed_signals": list_timed_signals(limit=limit_val)}
        return ActionResult(
            intention_key="MESSAGE_READY",
            payload={
                "message": "ok",
                "origin": "api",
                "channel_hint": "api",
                "correlation_id": correlation_id,
                "data": data,
                "audience": {"kind": "system", "id": "system"},
            },
            urgency="normal",
        )
