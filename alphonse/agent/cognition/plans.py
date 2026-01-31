from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PlanType(str, Enum):
    COMMUNICATE = "COMMUNICATE"
    SCHEDULE_TIMED_SIGNAL = "SCHEDULE_TIMED_SIGNAL"
    QUERY_STATUS = "QUERY_STATUS"
    SET_PREFERENCE = "SET_PREFERENCE"
    NOOP = "NOOP"


class CortexPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plan_type: PlanType
    priority: int = 0
    target: str | None = None
    channels: list[str] | None = None
    payload: dict[str, Any]
    requires_confirmation: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CortexResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply_text: str | None = None
    plans: list[CortexPlan] = Field(default_factory=list)
    cognition_state: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


class CommunicatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    style: str | None = None
    locale: str | None = None


class ScheduleTimedSignalPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal_type: str
    trigger_at: str
    timezone: str
    reminder_text: str
    origin: str
    chat_id: str
    correlation_id: str


class QueryStatusPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include: list[str]
