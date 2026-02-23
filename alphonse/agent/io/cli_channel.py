from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from alphonse.agent.io.contracts import NormalizedInboundMessage, NormalizedOutboundMessage
from alphonse.agent.io.adapters import SenseAdapter, ExtremityAdapter
from alphonse.agent.observability.log_manager import get_component_logger

logger = get_component_logger("io.cli_channel")


@dataclass(frozen=True)
class CliSenseAdapter(SenseAdapter):
    channel_type: str = "cli"

    def normalize(self, payload: dict[str, Any]) -> NormalizedInboundMessage:
        text = str(payload.get("text", "")).strip()
        user_name = _as_optional_str(payload.get("user_name"))
        timestamp = _as_float(payload.get("timestamp"), default=time.time())
        correlation_id = _as_optional_str(payload.get("correlation_id"))

        return NormalizedInboundMessage(
            text=text,
            channel_type=self.channel_type,
            channel_target=_as_optional_str(payload.get("target")) or "cli",
            user_id=None,
            user_name=user_name,
            timestamp=timestamp,
            correlation_id=correlation_id,
            metadata={"raw": payload},
        )


class CliExtremityAdapter(ExtremityAdapter):
    channel_type: str = "cli"

    def deliver(self, message: NormalizedOutboundMessage) -> None:
        if not message.message:
            return
        logger.info("CLI deliver text_len=%s correlation_id=%s", len(message.message), message.correlation_id)
        print(message.message)


def _as_optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_float(value: object | None, *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
