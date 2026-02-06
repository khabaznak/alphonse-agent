from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from alphonse.config import settings


class PlanningMode(str, Enum):
    AVENTURIZACION = "aventurizacion"
    CONTRATO_DE_RESULTADO = "contrato_de_resultado"


class PlanStepOptionKind(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    ALTERNATIVE = "alternative"
    ASK_CLARIFY = "ask_clarify"


class PlanFeedbackKind(str, Enum):
    STEP_CONFIRMED = "step_confirmed"
    STEP_REJECTED = "step_rejected"
    STEP_ALTERNATIVE_SELECTED = "step_alternative_selected"
    CRITERIA_EDITED = "criteria_edited"
    MODE_OVERRIDDEN = "mode_overridden"
    MODE_POLICY_ADJUSTED = "mode_policy_adjusted"


class PlanStepOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    option_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    kind: PlanStepOptionKind
    payload: dict[str, Any] = Field(default_factory=dict)
    order: int = 0


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    summary: str
    options: list[PlanStepOption] = Field(default_factory=list)
    selected_option_id: str | None = None


class AcceptanceCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    is_required: bool = True


class PlanFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    kind: PlanFeedbackKind
    step_id: str | None = None
    option_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    intent: str
    planning_mode: PlanningMode
    autonomy_level: float
    steps: list[PlanStep] = Field(default_factory=list)
    acceptance_criteria: list[AcceptanceCriterion] | None = None
    feedback_log: list[PlanFeedback] = Field(default_factory=list)
    usage_count: int = 0
    trust_score: float | None = None


@dataclass(frozen=True)
class PlanningContext:
    autonomy_level: float
    requested_mode: PlanningMode | None
    suggested_mode: PlanningMode
    effective_mode: PlanningMode
    user_override: bool


def normalize_autonomy_level(level: float | None) -> float:
    if level is None:
        return settings.get_autonomy_level()
    try:
        value = float(level)
    except (TypeError, ValueError):
        return settings.get_autonomy_level()
    return min(max(value, 0.0), 1.0)


def suggest_planning_mode(autonomy_level: float) -> PlanningMode:
    if autonomy_level < 0.34:
        return PlanningMode.AVENTURIZACION
    if autonomy_level < 0.74:
        return PlanningMode.CONTRATO_DE_RESULTADO
    return PlanningMode.CONTRATO_DE_RESULTADO


def resolve_planning_context(
    *,
    autonomy_level: float | None = None,
    requested_mode: PlanningMode | None = None,
) -> PlanningContext:
    resolved_autonomy = normalize_autonomy_level(autonomy_level)
    suggested = suggest_planning_mode(resolved_autonomy)
    if requested_mode is not None:
        return PlanningContext(
            autonomy_level=resolved_autonomy,
            requested_mode=requested_mode,
            suggested_mode=suggested,
            effective_mode=requested_mode,
            user_override=True,
        )
    configured_raw = settings.get_planning_mode()
    configured = _parse_planning_mode(configured_raw)
    if configured is not None:
        return PlanningContext(
            autonomy_level=resolved_autonomy,
            requested_mode=None,
            suggested_mode=suggested,
            effective_mode=configured,
            user_override=False,
        )
    return PlanningContext(
        autonomy_level=resolved_autonomy,
        requested_mode=None,
        suggested_mode=suggested,
        effective_mode=suggested,
        user_override=False,
    )


def _parse_planning_mode(value: str | None) -> PlanningMode | None:
    if not value:
        return None
    normalized = value.strip().lower()
    for mode in PlanningMode:
        if mode.value == normalized:
            return mode
    return None


def parse_planning_mode(value: str | None) -> PlanningMode | None:
    return _parse_planning_mode(value)
