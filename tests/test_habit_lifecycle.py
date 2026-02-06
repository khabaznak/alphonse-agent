from __future__ import annotations

import pytest

from alphonse.agent.cognition.habit_lifecycle import (
    HabitEvent,
    HabitEventType,
    HabitRecord,
    HabitState,
    HabitThresholds,
    TriggerType,
    create_habit,
    next_habit_state,
)
from alphonse.agent.cognition.intent_lifecycle import LifecycleState
from alphonse.agent.cognition.intent_registry import RiskLevel


def _record(state: HabitState, *, success: int = 0, failures: int = 0, executions: int = 0, opt_in: bool = True) -> HabitRecord:
    return HabitRecord(
        habit_id="habit",
        intent_signature_key="sig",
        trigger_type=TriggerType.TIME,
        trigger_definition="{}",
        target="self",
        lifecycle_state=state,
        autonomy_level_override=None,
        created_at="now",
        last_executed_at=None,
        execution_count=executions,
        success_count=success,
        failure_count=failures,
        paused=False,
        user_opt_in=opt_in,
        audit_required=True,
    )


def test_habit_creation_requires_opt_in() -> None:
    with pytest.raises(ValueError):
        create_habit(
            habit_id="h1",
            intent_signature_key="sig",
            trigger_type=TriggerType.TIME,
            trigger_definition={"cron": "* * * * *"},
            target="self",
            user_opt_in=False,
        )


def test_intent_lifecycle_gates_automation() -> None:
    thresholds = HabitThresholds()
    record = _record(HabitState.TRUSTED, success=6, failures=0, executions=6, opt_in=True)
    event = HabitEvent(HabitEventType.EXECUTION_SUCCEEDED, risk_level=RiskLevel.LOW, policy_allows=True)
    state = next_habit_state(record.lifecycle_state, event, record, thresholds, intent_state=LifecycleState.TRUSTED)
    assert state == HabitState.TRUSTED
    state = next_habit_state(record.lifecycle_state, event, record, thresholds, intent_state=LifecycleState.HABITUAL)
    assert state == HabitState.AUTOMATED


def test_regression_on_failures() -> None:
    thresholds = HabitThresholds()
    record = _record(HabitState.TRUSTED, success=5, failures=1, executions=6, opt_in=True)
    event = HabitEvent(HabitEventType.EXECUTION_FAILED)
    state = next_habit_state(record.lifecycle_state, event, record, thresholds, intent_state=LifecycleState.HABITUAL)
    assert state == HabitState.STABILIZING


def test_pause_resume_behavior() -> None:
    thresholds = HabitThresholds()
    record = _record(HabitState.ACTIVE)
    paused = next_habit_state(record.lifecycle_state, HabitEvent(HabitEventType.USER_PAUSED), record, thresholds, intent_state=LifecycleState.TRUSTED)
    assert paused == HabitState.PAUSED
    resumed = next_habit_state(paused, HabitEvent(HabitEventType.USER_RESUMED), record, thresholds, intent_state=LifecycleState.TRUSTED)
    assert resumed == HabitState.ACTIVE
