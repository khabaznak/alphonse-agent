from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionResult:
    intention_key: str
    payload: dict[str, Any]
    urgency: str | None
    requires_narration: bool = False
    narration_style: str | None = None
