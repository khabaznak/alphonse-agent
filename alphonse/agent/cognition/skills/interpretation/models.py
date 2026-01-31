from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class MessageEvent:
    text: str
    user_id: str | None
    channel: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingDecision:
    skill: str
    args: dict[str, Any]
    confidence: float
    needs_clarification: bool
    clarifying_question: str | None
    path: Literal["deterministic", "llm"]
