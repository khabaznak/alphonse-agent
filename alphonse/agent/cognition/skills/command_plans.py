from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ActorChannel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    target: str


class Actor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_id: str | None = None
    channel: ActorChannel


class IntentEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reminder_verbs: list[str] = Field(default_factory=list)
    time_hints: list[str] = Field(default_factory=list)
    quoted_spans: list[str] = Field(default_factory=list)
    score: float = 0.0


class CommandPlanBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_kind: str
    plan_version: int
    plan_id: str
    correlation_id: str
    created_at: str
    source: str
    actor: Actor
    intent_confidence: float
    requires_confirmation: bool = False
    questions: list[str] = Field(default_factory=list)
    intent_evidence: IntentEvidence
    payload: dict
    metadata: dict | None = None
    original_text: str | None = None


class GreetingPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str | None = None
    text: str | None = None


class UnknownPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_text: str
    reason: str
    suggestions: list[str] = Field(default_factory=list)


class PersonRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["person_id", "alias_name"]
    id: str | None = None
    name: str | None = None


class TargetRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_ref: PersonRef


class ReminderSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timezone: str
    trigger_at: str | None = None
    rrule: str | None = None
    time_of_day: Literal["morning", "afternoon", "evening"] | None = None


class ReminderMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str | None = None
    text: str


class ReminderDelivery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel_type: str | None = None
    priority: Literal["low", "normal", "high", "urgent"] | None = None


class CreateReminderPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: TargetRef
    schedule: ReminderSchedule
    message: ReminderMessage
    delivery: ReminderDelivery | None = None
    idempotency_key: str | None = None


class SendMessagePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: TargetRef
    message: ReminderMessage
    delivery: ReminderDelivery | None = None


class GreetingPlan(CommandPlanBase):
    plan_kind: Literal["greeting"]
    plan_version: int = 1
    payload: GreetingPayload


class UnknownPlan(CommandPlanBase):
    plan_kind: Literal["unknown"]
    plan_version: int = 1
    payload: UnknownPayload


class CreateReminderPlan(CommandPlanBase):
    plan_kind: Literal["create_reminder"]
    plan_version: int = 1
    payload: CreateReminderPayload
    intent_evidence: IntentEvidence


class SendMessagePlan(CommandPlanBase):
    plan_kind: Literal["send_message"]
    plan_version: int = 1
    payload: SendMessagePayload
    intent_evidence: IntentEvidence


CommandPlan = GreetingPlan | UnknownPlan | CreateReminderPlan | SendMessagePlan


def parse_command_plan(data: dict) -> CommandPlan:
    kind = data.get("plan_kind")
    if kind == "greeting":
        return GreetingPlan.model_validate(data)
    if kind == "unknown":
        return UnknownPlan.model_validate(data)
    if kind == "create_reminder":
        return CreateReminderPlan.model_validate(data)
    if kind == "send_message":
        return SendMessagePlan.model_validate(data)
    raise ValueError(f"Unknown plan kind: {kind}")
