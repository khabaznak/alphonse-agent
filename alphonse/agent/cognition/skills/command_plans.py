from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PlanBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    kind: str
    version: int = 1
    created_at: str
    created_by: str | None
    source: str
    correlation_id: str
    original_text: str | None = None


class ReminderSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timezone: str
    trigger_at: str | None = None
    rrule: str | None = None


class ReminderMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str
    text: str


class ReminderDelivery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferred_channel_type: str | None = None
    priority: Literal["low", "normal", "high", "urgent"] = "normal"


class CreateReminderPlan(PlanBase):
    kind: Literal["create_reminder"]
    target_person_id: str | None = None
    schedule: ReminderSchedule
    message: ReminderMessage
    delivery: ReminderDelivery


class SendMessagePlan(PlanBase):
    kind: Literal["send_message"]
    target_person_id: str | None = None
    message: ReminderMessage
    channel_type: str | None = None


CommandPlan = CreateReminderPlan | SendMessagePlan


def parse_plan(data: dict) -> CommandPlan:
    kind = data.get("kind")
    if kind == "create_reminder":
        return CreateReminderPlan.model_validate(data)
    if kind == "send_message":
        return SendMessagePlan.model_validate(data)
    raise ValueError(f"Unknown plan kind: {kind}")
