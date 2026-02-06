from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    skill: str
    args: dict[str, Any] = Field(default_factory=dict)
    expects_receipt: bool = True
    retry_policy: dict[str, Any] | None = None


class Stage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_id: str
    t_plus_sec: int
    actions: list[Action]


class CommandPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    plan_type: str
    trigger: str
    version: int
    created_at: str
    ttl_sec: int
    default_on_timeout: str
    cancel_on_resolution: bool
    stages: list[Stage]


def build_plan(
    *,
    plan_type: str,
    trigger: str,
    ttl_sec: int,
    default_on_timeout: str,
    cancel_on_resolution: bool,
    stages: list[Stage],
    version: int = 1,
) -> CommandPlan:
    return CommandPlan(
        plan_id=str(uuid4()),
        plan_type=plan_type,
        trigger=trigger,
        version=version,
        created_at=datetime.now(timezone.utc).isoformat(),
        ttl_sec=ttl_sec,
        default_on_timeout=default_on_timeout,
        cancel_on_resolution=cancel_on_resolution,
        stages=stages,
    )
