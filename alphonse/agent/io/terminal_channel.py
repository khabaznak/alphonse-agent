from __future__ import annotations

from alphonse.agent.io.contracts import NormalizedOutboundMessage
from alphonse.agent.observability.log_manager import get_component_logger

logger = get_component_logger("io.terminal_channel")


class TerminalExtremityAdapter:
    channel_type = "terminal"

    def deliver(self, message: NormalizedOutboundMessage) -> None:
        logger.info(
            "TerminalExtremityAdapter delivered target=%s correlation_id=%s",
            message.channel_target,
            message.correlation_id,
        )
