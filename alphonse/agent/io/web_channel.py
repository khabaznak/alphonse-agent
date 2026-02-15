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
        text = _extract_inbound_text(payload)
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
        from alphonse.infrastructure.web_event_hub import web_event_hub

        # API/web UI currently use the API exchange for request-response delivery.
        event_payload = {
            "message": message.message,
            "data": (message.metadata or {}).get("data"),
            "channel_target": message.channel_target,
            "correlation_id": message.correlation_id,
        }
        if message.channel_target:
            web_event_hub.publish(str(message.channel_target), event_payload)
        if message.correlation_id and gateway.exchange:
            gateway.exchange.publish(
                str(message.correlation_id),
                {"message": message.message, "data": (message.metadata or {}).get("data")},
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


def _extract_inbound_text(payload: dict[str, Any]) -> str:
    text = str(payload.get("text") or "").strip()
    if text:
        return text
    content = payload.get("content")
    if not isinstance(content, dict):
        return ""
    content_type = str(content.get("type") or "").strip().lower()
    if content_type == "text":
        return str(content.get("text") or "").strip()
    if content_type == "asset":
        assets = content.get("assets")
        if isinstance(assets, list):
            for item in assets:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("kind") or "").strip().lower()
                if kind == "audio":
                    return "[audio asset message]"
        return "[asset message]"
    return ""
