from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ResponseKind = Literal["greeting", "ack", "clarify", "answer", "policy_block", "error"]


@dataclass(frozen=True)
class ResponseSpec:
    kind: ResponseKind
    key: str
    locale: str | None = None
    address_style: str | None = None
    tone: str | None = None
    variables: dict[str, Any] = field(default_factory=dict)
    options: list[dict[str, Any]] | None = None
    next_prompt: str | None = None
