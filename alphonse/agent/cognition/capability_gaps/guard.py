from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from alphonse.agent.cognition.intent_registry import IntentCategory


class PlanStatus(str, Enum):
    PLANNING = "planning"
    AWAITING_USER = "awaiting_user"
    NEEDS_CONFIRMATION = "needs_confirmation"
    EXECUTING = "executing"
    FAILED = "failed"


@dataclass(frozen=True)
class GapGuardInput:
    category: IntentCategory | None
    plan_status: PlanStatus | None
    needs_clarification: bool
    reason: str | None


def should_create_gap(data: GapGuardInput) -> bool:
    if data.category == IntentCategory.CORE_CONVERSATIONAL:
        return False
    if data.category == IntentCategory.CONTROL_PLANE:
        return False
    if data.plan_status in {
        PlanStatus.PLANNING,
        PlanStatus.AWAITING_USER,
        PlanStatus.NEEDS_CONFIRMATION,
    }:
        return False
    if data.needs_clarification:
        return False
    if data.category != IntentCategory.TASK_PLANE:
        return False
    return data.reason == "missing_capability"
