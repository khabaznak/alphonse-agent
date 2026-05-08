from __future__ import annotations

from dataclasses import dataclass

from alphonse.agent.io.contracts import NormalizedOutboundMessage
from alphonse.agent.io.adapters import ExtremityAdapter
from alphonse.agent.observability.log_manager import get_component_logger

logger = get_component_logger("io.cli_channel")


class CliExtremityAdapter(ExtremityAdapter):
    channel_type: str = "cli"

    def deliver(self, message: NormalizedOutboundMessage) -> None:
        if not message.message:
            return
        logger.info("CLI deliver text_len=%s correlation_id=%s", len(message.message), message.correlation_id)
        print(message.message)
