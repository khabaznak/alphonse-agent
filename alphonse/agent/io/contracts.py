from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NormalizedInboundMessage:
    text: str
    channel_type: str
    channel_target: str | None
    user_id: str | None
    user_name: str | None
    timestamp: float
    correlation_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedOutboundMessage:
    message: str
    channel_type: str
    channel_target: str | None
    audience: dict[str, str]
    correlation_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
