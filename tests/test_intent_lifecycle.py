from __future__ import annotations

from alphonse.agent.cognition.intent_lifecycle import (
    LifecycleEvent,
    LifecycleEventType,
    LifecycleHint,
    LifecycleRecord,
    LifecycleState,
    LifecycleThresholds,
    lifecycle_hint,
    next_state,
)
from alphonse.agent.cognition.intent_types import IntentCategory
from alphonse.agent.cognition.planning import PlanningMode
from alphonse.agent.cognition.planning_engine import propose_plan


def _record(state: LifecycleState, *, success: int = 0, corrections: int = 0, usage: int = 0, opt_in: bool = False) -> LifecycleRecord:
    return LifecycleRecord(
        signature_key="sig",
        intent_name="schedule_reminder",
        category=IntentCategory.TASK_PLANE,
        state=state,
        first_seen_at="now",
        last_seen_at="now",
        usage_count=usage,
        success_count=success,
        correction_count=corrections,
        last_mode_used=None,
        last_outcome=None,
        trust_score=None,
        opt_in_automated=opt_in,
    )


def test_progression_discovered_to_guided_to_trusted() -> None:
    thresholds = LifecycleThresholds(trust_success_count=3, habitual_success_count=5)
    record = _record(LifecycleState.DISCOVERED, success=0, corrections=0, usage=0)
    state = next_state(
        record.state,
        LifecycleEvent(LifecycleEventType.INTENT_DETECTED, recognized=True),
        record,
        thresholds,
    )
    assert state == LifecycleState.BOOTSTRAPPED
    record.state = state
    record.success_count = 1
    record.usage_count = 1
    state = next_state(
        record.state,
        LifecycleEvent(LifecycleEventType.EXECUTION_SUCCEEDED),
        record,
        thresholds,
    )
    assert state == LifecycleState.GUIDED
    record.state = state
    record.success_count = 3
    record.usage_count = 3
    state = next_state(
        record.state,
        LifecycleEvent(LifecycleEventType.EXECUTION_SUCCEEDED),
        record,
        thresholds,
    )
    assert state == LifecycleState.TRUSTED


def test_regression_on_corrections() -> None:
    thresholds = LifecycleThresholds()
    record = _record(LifecycleState.TRUSTED, success=3, corrections=1, usage=3)
    state = next_state(
        record.state,
        LifecycleEvent(LifecycleEventType.USER_CORRECTED_PLAN),
        record,
        thresholds,
    )
    assert state == LifecycleState.GUIDED


def test_automation_requires_opt_in() -> None:
    thresholds = LifecycleThresholds(trust_success_count=3, habitual_success_count=5)
    record = _record(LifecycleState.HABITUAL, success=6, corrections=0, usage=6, opt_in=False)
    state = next_state(
        record.state,
        LifecycleEvent(LifecycleEventType.EXECUTION_SUCCEEDED),
        record,
        thresholds,
    )
    assert state == LifecycleState.HABITUAL
    record.opt_in_automated = True
    state = next_state(
        record.state,
        LifecycleEvent(LifecycleEventType.EXECUTION_SUCCEEDED),
        record,
        thresholds,
    )
    assert state == LifecycleState.AUTOMATED


def test_lifecycle_hint_affects_planning_mode() -> None:
    hint = lifecycle_hint(LifecycleState.DISCOVERED, IntentCategory.TASK_PLANE)
    proposal = propose_plan(
        intent="schedule_reminder",
        autonomy_level=0.9,
        requested_mode=None,
        draft_steps=["Do thing"],
        acceptance_criteria=["Done"],
        lifecycle_hint=hint,
    )
    assert proposal.plan.planning_mode == PlanningMode.AVENTURIZACION
    assert proposal.plan.autonomy_level <= hint.autonomy_cap
