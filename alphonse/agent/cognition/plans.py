from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

class CortexPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0
    target: str | None = None
    channels: list[str] | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _normalize_shape(self) -> "CortexPlan":
        if not self.parameters and self.payload:
            self.parameters = dict(self.payload)
        if self.parameters and not self.payload:
            self.payload = dict(self.parameters)
        self.tool = str(self.tool or "").strip().lower()
        return self


class CortexResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply_text: str | None = None
    plans: list[CortexPlan] = Field(default_factory=list)
    cognition_state: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)
