from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from alphonse.agent.io.contracts import NormalizedInboundMessage, NormalizedOutboundMessage
from alphonse.agent.io.adapters import SenseAdapter, ExtremityAdapter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebSenseAdapter(SenseAdapter):
    channel_type: str = "webui"

    def normalize(self, payload: dict[str, Any]) -> NormalizedInboundMessage:
        text = str(payload.get("text", "")).strip()
        user_id = _as_optional_str(payload.get("user_id"))
        user_name = _as_optional_str(payload.get("user_name"))
        timestamp = _as_float(payload.get("timestamp"), default=time.time())
        correlation_id = _as_optional_str(payload.get("correlation_id"))

        return NormalizedInboundMessage(
            text=text,
            channel_type=_as_optional_str(payload.get("channel")) or self.channel_type,
            channel_target=_as_optional_str(payload.get("target")) or "webui",
            user_id=user_id,
            user_name=user_name,
            timestamp=timestamp,
            correlation_id=correlation_id,
            metadata={"raw": payload},
        )


class WebExtremityAdapter(ExtremityAdapter):
    channel_type: str = "webui"

    def deliver(self, message: NormalizedOutboundMessage) -> None:
        from alphonse.infrastructure.api_gateway import gateway

        # API/web UI currently use the API exchange for request-response delivery.
        if message.correlation_id and gateway.exchange:
            gateway.exchange.publish(
                str(message.correlation_id),
                {
                    "message": message.message,
                    "data": (message.metadata or {}).get("data"),
                },
            )
        logger.info(
            "WebExtremityAdapter delivered target=%s correlation_id=%s",
            message.channel_target,
            message.correlation_id,
        )


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
