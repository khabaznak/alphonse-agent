from __future__ import annotations

import logging
import time
from typing import Any

from alphonse.agent.io.contracts import NormalizedInboundMessage, NormalizedOutboundMessage

logger = logging.getLogger(__name__)


class TerminalSenseAdapter:
    channel_type = "terminal"

    def normalize(self, payload: dict[str, Any]) -> NormalizedInboundMessage:
        text = str(payload.get("text") or "").strip()
        correlation_id = payload.get("correlation_id")
        return NormalizedInboundMessage(
            text=text,
            channel_type="terminal",
            channel_target=str(payload.get("channel_target") or "terminal"),
            user_id=str(payload.get("user_id") or "terminal"),
            user_name=str(payload.get("user_name") or "terminal"),
            timestamp=float(payload.get("timestamp") or time.time()),
            correlation_id=str(correlation_id) if correlation_id else None,
            metadata=dict(payload.get("metadata") or {}),
        )


class TerminalExtremityAdapter:
    channel_type = "terminal"

    def deliver(self, message: NormalizedOutboundMessage) -> None:
        logger.info(
            "TerminalExtremityAdapter delivered target=%s correlation_id=%s",
            message.channel_target,
            message.correlation_id,
        )
