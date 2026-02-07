from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from alphonse.agent.cognition.intent_lifecycle import LifecycleState, get_record
from alphonse.agent.cognition.intent_types import IntentCategory, RiskLevel
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

logger = logging.getLogger(__name__)


class HabitState(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    ACTIVE = "active"
    STABILIZING = "stabilizing"
    TRUSTED = "trusted"
    AUTOMATED = "automated"
    PAUSED = "paused"
    RETIRED = "retired"


class HabitEventType(str, Enum):
    HABIT_PROPOSED = "habit_proposed"
    HABIT_ACCEPTED = "habit_accepted"
    HABIT_REJECTED = "habit_rejected"
    EXECUTION_SUCCEEDED = "execution_succeeded"
    EXECUTION_FAILED = "execution_failed"
    USER_CORRECTED = "user_corrected"
    USER_PAUSED = "user_paused"
    USER_RESUMED = "user_resumed"
    USER_RETIRED = "user_retired"
    POLICY_BLOCKED = "policy_blocked"


class TriggerType(str, Enum):
    TIME = "time"
    EVENT = "event"
    CONDITION = "condition"


@dataclass
class HabitRecord:
    habit_id: str
    intent_signature_key: str
    trigger_type: TriggerType
    trigger_definition: str
    target: str
    lifecycle_state: HabitState
    autonomy_level_override: float | None
    created_at: str
    last_executed_at: str | None
    execution_count: int
    success_count: int
    failure_count: int
    paused: bool
    user_opt_in: bool
    audit_required: bool


@dataclass(frozen=True)
class HabitEvent:
    event_type: HabitEventType
    risk_level: RiskLevel = RiskLevel.LOW
    policy_allows: bool = True


@dataclass(frozen=True)
class HabitThresholds:
    stabilizing_success_count: int = 2
    trusted_success_count: int = 5
    correction_rate_threshold: float = 0.2


@dataclass(frozen=True)
class HabitLifecycleHint:
    confirmation_required: bool
    autonomy_cap: float
    audit_level: str


def create_habit(
    *,
    habit_id: str,
    intent_signature_key: str,
    trigger_type: TriggerType,
    trigger_definition: dict[str, Any],
    target: str,
    user_opt_in: bool,
    audit_required: bool = True,
    autonomy_level_override: float | None = None,
) -> HabitRecord:
    if not user_opt_in:
        raise ValueError("Habit requires explicit user opt-in")
    if not intent_signature_key:
        raise ValueError("Habit requires an intent signature key")
    record = HabitRecord(
        habit_id=habit_id,
        intent_signature_key=intent_signature_key,
        trigger_type=trigger_type,
        trigger_definition=json.dumps(trigger_definition, sort_keys=True),
        target=target,
        lifecycle_state=HabitState.PROPOSED,
        autonomy_level_override=autonomy_level_override,
        created_at=_now(),
        last_executed_at=None,
        execution_count=0,
        success_count=0,
        failure_count=0,
        paused=False,
        user_opt_in=user_opt_in,
        audit_required=audit_required,
    )
    upsert_habit(record)
    return record


def next_habit_state(
    current: HabitState,
    event: HabitEvent,
    record: HabitRecord,
    thresholds: HabitThresholds,
    *,
    intent_state: LifecycleState | None,
) -> HabitState:
    if event.event_type == HabitEventType.HABIT_PROPOSED:
        return HabitState.PROPOSED
    if event.event_type == HabitEventType.HABIT_ACCEPTED:
        return HabitState.ACCEPTED
    if event.event_type == HabitEventType.HABIT_REJECTED:
        return HabitState.RETIRED
    if event.event_type == HabitEventType.USER_RETIRED:
        return HabitState.RETIRED
    if event.event_type in {HabitEventType.USER_PAUSED, HabitEventType.POLICY_BLOCKED}:
        return HabitState.PAUSED
    if event.event_type == HabitEventType.USER_RESUMED:
        return HabitState.ACTIVE
    if event.event_type == HabitEventType.EXECUTION_FAILED:
        if current in {HabitState.TRUSTED, HabitState.AUTOMATED}:
            return HabitState.STABILIZING
        return current
    if event.event_type == HabitEventType.USER_CORRECTED:
        if current in {HabitState.TRUSTED, HabitState.AUTOMATED}:
            return HabitState.STABILIZING
        return current
    if event.event_type == HabitEventType.EXECUTION_SUCCEEDED:
        if current == HabitState.ACCEPTED:
            return HabitState.ACTIVE
        if current == HabitState.ACTIVE and record.success_count >= thresholds.stabilizing_success_count:
            return HabitState.STABILIZING
        if current == HabitState.STABILIZING and record.success_count >= thresholds.trusted_success_count:
            if _correction_rate(record) <= thresholds.correction_rate_threshold:
                return HabitState.TRUSTED
        if current == HabitState.TRUSTED:
            if _can_automate(record, event, intent_state):
                return HabitState.AUTOMATED
    return current


def record_habit_event(
    habit_id: str,
    event: HabitEvent,
    *,
    thresholds: HabitThresholds | None = None,
) -> HabitRecord:
    thresholds = thresholds or HabitThresholds()
    record = get_habit(habit_id)
    if record is None:
        raise ValueError("Habit not found")
    intent_state = _intent_state_for(record.intent_signature_key)
    record.execution_count += 1
    if event.event_type == HabitEventType.EXECUTION_SUCCEEDED:
        record.success_count += 1
        record.last_executed_at = _now()
    if event.event_type == HabitEventType.EXECUTION_FAILED:
        record.failure_count += 1
        record.last_executed_at = _now()
    if event.event_type in {HabitEventType.USER_PAUSED, HabitEventType.POLICY_BLOCKED}:
        record.paused = True
    if event.event_type == HabitEventType.USER_RESUMED:
        record.paused = False
    new_state = next_habit_state(record.lifecycle_state, event, record, thresholds, intent_state=intent_state)
    if new_state != record.lifecycle_state:
        _emit_state_change(record, new_state, event)
        record.lifecycle_state = new_state
    upsert_habit(record)
    return record


def habit_lifecycle_hint(habit: HabitRecord) -> HabitLifecycleHint:
    if habit.lifecycle_state in {HabitState.PROPOSED, HabitState.ACCEPTED}:
        return HabitLifecycleHint(confirmation_required=True, autonomy_cap=0.3, audit_level="full")
    if habit.lifecycle_state == HabitState.ACTIVE:
        return HabitLifecycleHint(confirmation_required=True, autonomy_cap=0.5, audit_level="full")
    if habit.lifecycle_state == HabitState.STABILIZING:
        return HabitLifecycleHint(confirmation_required=True, autonomy_cap=0.6, audit_level="full")
    if habit.lifecycle_state == HabitState.TRUSTED:
        return HabitLifecycleHint(confirmation_required=False, autonomy_cap=0.8, audit_level="full")
    if habit.lifecycle_state == HabitState.AUTOMATED:
        return HabitLifecycleHint(confirmation_required=False, autonomy_cap=1.0, audit_level="full")
    return HabitLifecycleHint(confirmation_required=True, autonomy_cap=0.0, audit_level="full")


def describe_habit(habit_id: str) -> dict[str, Any]:
    habit = get_habit(habit_id)
    if habit is None:
        return {"habit_id": habit_id, "status": "not_found"}
    return {
        "habit_id": habit.habit_id,
        "intent_signature_key": habit.intent_signature_key,
        "trigger": {
            "type": habit.trigger_type.value,
            "definition": habit.trigger_definition,
        },
        "target": habit.target,
        "state": habit.lifecycle_state.value,
        "execution_count": habit.execution_count,
        "success_count": habit.success_count,
        "failure_count": habit.failure_count,
        "last_executed_at": habit.last_executed_at,
        "paused": habit.paused,
    }


def get_habit(habit_id: str) -> HabitRecord | None:
    db_path = resolve_nervous_system_db_path()
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT habit_id, intent_signature_key, trigger_type, trigger_definition, target,
                       lifecycle_state, autonomy_level_override, created_at, last_executed_at,
                       execution_count, success_count, failure_count, paused, user_opt_in, audit_required
                FROM habit_lifecycle
                WHERE habit_id = ?
                """,
                (habit_id,),
            ).fetchone()
    except sqlite3.OperationalError:
        return None
    if not row:
        return None
    return HabitRecord(
        habit_id=row[0],
        intent_signature_key=row[1],
        trigger_type=TriggerType(row[2]),
        trigger_definition=row[3],
        target=row[4],
        lifecycle_state=HabitState(row[5]),
        autonomy_level_override=row[6],
        created_at=row[7],
        last_executed_at=row[8],
        execution_count=int(row[9]),
        success_count=int(row[10]),
        failure_count=int(row[11]),
        paused=bool(row[12]),
        user_opt_in=bool(row[13]),
        audit_required=bool(row[14]),
    )


def upsert_habit(record: HabitRecord) -> None:
    db_path = resolve_nervous_system_db_path()
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO habit_lifecycle
                  (habit_id, intent_signature_key, trigger_type, trigger_definition, target,
                   lifecycle_state, autonomy_level_override, created_at, last_executed_at,
                   execution_count, success_count, failure_count, paused, user_opt_in, audit_required)
                VALUES
                  (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(habit_id) DO UPDATE SET
                  intent_signature_key=excluded.intent_signature_key,
                  trigger_type=excluded.trigger_type,
                  trigger_definition=excluded.trigger_definition,
                  target=excluded.target,
                  lifecycle_state=excluded.lifecycle_state,
                  autonomy_level_override=excluded.autonomy_level_override,
                  last_executed_at=excluded.last_executed_at,
                  execution_count=excluded.execution_count,
                  success_count=excluded.success_count,
                  failure_count=excluded.failure_count,
                  paused=excluded.paused,
                  user_opt_in=excluded.user_opt_in,
                  audit_required=excluded.audit_required
                """,
                (
                    record.habit_id,
                    record.intent_signature_key,
                    record.trigger_type.value,
                    record.trigger_definition,
                    record.target,
                    record.lifecycle_state.value,
                    record.autonomy_level_override,
                    record.created_at,
                    record.last_executed_at,
                    record.execution_count,
                    record.success_count,
                    record.failure_count,
                    1 if record.paused else 0,
                    1 if record.user_opt_in else 0,
                    1 if record.audit_required else 0,
                ),
            )
            conn.commit()
    except sqlite3.OperationalError:
        return


def _intent_state_for(signature_key_value: str) -> LifecycleState | None:
    record = get_record(signature_key_value)
    return record.state if record else None


def _correction_rate(record: HabitRecord) -> float:
    if record.execution_count <= 0:
        return 0.0
    return record.failure_count / record.execution_count


def _can_automate(
    record: HabitRecord,
    event: HabitEvent,
    intent_state: LifecycleState | None,
) -> bool:
    if event.risk_level != RiskLevel.LOW:
        return False
    if not record.user_opt_in:
        return False
    if not event.policy_allows:
        return False
    if intent_state is None:
        return False
    return intent_state in {LifecycleState.HABITUAL, LifecycleState.AUTOMATED}


def _emit_state_change(record: HabitRecord, new_state: HabitState, event: HabitEvent) -> None:
    payload = {
        "habit_id": record.habit_id,
        "from_state": record.lifecycle_state.value,
        "to_state": new_state.value,
        "reason": event.event_type.value,
        "timestamp": _now(),
    }
    logger.info("habit.state_changed %s", payload)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
