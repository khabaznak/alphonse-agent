from __future__ import annotations

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.observability.log_manager import get_component_logger

logger = get_component_logger("actions.handle_subconscious_prompt")


class HandleSubconsciousPromptAction(Action):
    key = "handle_subconscious_prompt"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        payload = getattr(signal, "payload", {}) if signal else {}
        logger.info(
            "HandleSubconsciousPromptAction processed correlation_id=%s has_text=%s",
            getattr(signal, "correlation_id", None) if signal else None,
            bool(str((payload or {}).get("text") or "").strip()) if isinstance(payload, dict) else False,
        )
        return ActionResult(intention_key="NOOP", payload={}, urgency=None)
