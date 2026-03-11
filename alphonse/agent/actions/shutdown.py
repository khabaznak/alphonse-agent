from __future__ import annotations

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.nervous_system.senses.bus import Signal
from alphonse.agent.observability.log_manager import get_component_logger

logger = get_component_logger("actions.shutdown")


class ShutdownAction(Action):
    key = "shutdown"

    def execute(self, context: dict) -> ActionResult:
        bus = context.get("ctx")
        if hasattr(bus, "emit"):
            bus.emit(Signal(type="SHUTDOWN", payload={}, source="shutdown_action"))
        logger.info("ShutdownAction emitted SHUTDOWN signal")
        return ActionResult(intention_key="NOOP", payload={"shutdown": True}, urgency=None)
