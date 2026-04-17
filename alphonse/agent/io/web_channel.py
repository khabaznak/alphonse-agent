from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from alphonse.agent.io.contracts import NormalizedInboundMessage
from alphonse.agent.io.adapters import SenseAdapter


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
