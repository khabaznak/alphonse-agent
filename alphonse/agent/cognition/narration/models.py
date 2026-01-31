from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AudienceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["person", "group", "system"]
    id: str


class NarrationIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    should_narrate: bool
    audience: AudienceRef
    channel_type: str
    priority: Literal["low", "normal", "high", "urgent"]
    timing: Literal["now", "defer", "batch"]
    verbosity: Literal["terse", "brief", "normal", "verbose"]
    format: Literal["plain", "markdown", "html"]
    reason: str


class PresentationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str
    tone: str
    formality: str
    emoji: str
    verbosity_cap: str
    safety_mode: Literal["strict", "normal"]
    reason: str


class ModelPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["local", "openai"]
    model: str
    max_tokens: int
    temperature: float
    timeout_ms: int
    fallback: list[dict[str, str]] = Field(default_factory=list)
    reason: str


class MessageDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correlation_id: str
    audience: AudienceRef
    channel_type: str
    format: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RenderedMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel_type: str
    payload: dict[str, Any]


class ContextBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: dict[str, Any]
    trace: dict[str, Any]
    presence: dict[str, Any]
    time_context: dict[str, Any]
    channel_availability: dict[str, Any]
    identity: dict[str, Any]
