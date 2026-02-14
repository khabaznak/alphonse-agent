from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EmitStateDeliverable:
    type: Literal["emit_state"] = "emit_state"
    kind: str = "typing"


@dataclass(frozen=True)
class TextDeliverable:
    type: Literal["text"] = "text"
    text: str = ""


Deliverable = EmitStateDeliverable | TextDeliverable


@dataclass(frozen=True)
class RenderResult:
    status: Literal["rendered", "failed"]
    deliverables: list[Deliverable]
    error: str | None = None
