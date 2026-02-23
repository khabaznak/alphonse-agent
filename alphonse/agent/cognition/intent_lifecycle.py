from __future__ import annotations

import json
from alphonse.agent.observability.log_manager import get_component_logger
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from alphonse.agent.cognition.intent_types import IntentCategory
from alphonse.agent.cognition.planning import PlanningMode
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

logger = get_component_logger("cognition.intent_lifecycle")


class LifecycleState(str, Enum):
    DISCOVERED = "discovered"
    BOOTSTRAPPED = "bootstrapped"
    GUIDED = "guided"
    TRUSTED = "trusted"
    HABITUAL = "habitual"
    AUTOMATED = "automated"


class LifecycleEventType(str, Enum):
    INTENT_DETECTED = "intent_detected"
    PLAN_CONFIRMED = "plan_confirmed"
    EXECUTION_SUCCEEDED = "execution_succeeded"
    EXECUTION_FAILED = "execution_failed"
    USER_CORRECTED_PLAN = "user_corrected_plan"
    USER_ABORTED = "user_aborted"
    POLICY_BLOCKED = "policy_blocked"
    MODE_TRANSITION = "mode_transition"


@dataclass(frozen=True)
class IntentSignature:
    intent_name: str
    category: IntentCategory
    slots: dict[str, Any]
    user_scope: str | None = None


@dataclass
class LifecycleRecord:
    signature_key: str
    intent_name: str
    category: IntentCategory
    state: LifecycleState
    first_seen_at: str
    last_seen_at: str
    usage_count: int
    success_count: int
    correction_count: int
    last_mode_used: str | None
    last_outcome: str | None
    trust_score: float | None
    opt_in_automated: bool


@dataclass(frozen=True)
class LifecycleEvent:
    event_type: LifecycleEventType
    mode_used: PlanningMode | None = None
    outcome: str | None = None
    recognized: bool | None = None


@dataclass(frozen=True)
class LifecycleThresholds:
    trust_success_count: int = 3
    habitual_success_count: int = 5
    correction_rate_threshold: float = 0.2


@dataclass(frozen=True)
class LifecycleHint:
    preferred_mode: PlanningMode
    min_confirmation: bool
    autonomy_cap: float


def signature_key(signature: IntentSignature) -> str:
    slots_payload = _stable_json(signature.slots)
    scope = signature.user_scope or "global"
    return f"{scope}|{signature.intent_name}|{signature.category.value}|{slots_payload}"


def lifecycle_hint(state: LifecycleState, category: IntentCategory) -> LifecycleHint:
    if category == IntentCategory.CORE_CONVERSATIONAL:
        return LifecycleHint(
            preferred_mode=PlanningMode.AVENTURIZACION,
            min_confirmation=False,
            autonomy_cap=0.3,
        )
    if state in {LifecycleState.DISCOVERED, LifecycleState.BOOTSTRAPPED}:
        return LifecycleHint(
            preferred_mode=PlanningMode.AVENTURIZACION,
            min_confirmation=True,
            autonomy_cap=0.3,
        )
    if state == LifecycleState.GUIDED:
        return LifecycleHint(
            preferred_mode=PlanningMode.AVENTURIZACION,
            min_confirmation=True,
            autonomy_cap=0.5,
        )
    if state == LifecycleState.TRUSTED:
        return LifecycleHint(
            preferred_mode=PlanningMode.CONTRATO_DE_RESULTADO,
            min_confirmation=True,
            autonomy_cap=0.7,
        )
    if state == LifecycleState.HABITUAL:
        return LifecycleHint(
            preferred_mode=PlanningMode.CONTRATO_DE_RESULTADO,
            min_confirmation=False,
            autonomy_cap=0.85,
        )
    return LifecycleHint(
        preferred_mode=PlanningMode.CONTRATO_DE_RESULTADO,
        min_confirmation=False,
        autonomy_cap=1.0,
    )


def next_state(
    current: LifecycleState,
    event: LifecycleEvent,
    record: LifecycleRecord,
    thresholds: LifecycleThresholds,
) -> LifecycleState:
    if event.event_type == LifecycleEventType.INTENT_DETECTED:
        if event.recognized:
            return LifecycleState.BOOTSTRAPPED
        return LifecycleState.DISCOVERED
    if event.event_type == LifecycleEventType.POLICY_BLOCKED:
        return LifecycleState.GUIDED
    if event.event_type == LifecycleEventType.USER_CORRECTED_PLAN:
        if current in {LifecycleState.TRUSTED, LifecycleState.HABITUAL}:
            return LifecycleState.GUIDED
        return current
    if event.event_type == LifecycleEventType.EXECUTION_FAILED:
        if current in {LifecycleState.TRUSTED, LifecycleState.HABITUAL, LifecycleState.AUTOMATED}:
            return LifecycleState.GUIDED
        return current
    if event.event_type == LifecycleEventType.PLAN_CONFIRMED:
        if event.mode_used == PlanningMode.AVENTURIZACION and current in {
            LifecycleState.BOOTSTRAPPED,
            LifecycleState.DISCOVERED,
        }:
            return LifecycleState.GUIDED
    if event.event_type == LifecycleEventType.EXECUTION_SUCCEEDED:
        if current in {LifecycleState.BOOTSTRAPPED, LifecycleState.DISCOVERED}:
            return LifecycleState.GUIDED
        correction_rate = _correction_rate(record)
        if current == LifecycleState.GUIDED and record.success_count >= thresholds.trust_success_count:
            if correction_rate <= thresholds.correction_rate_threshold:
                return LifecycleState.TRUSTED
        if current == LifecycleState.TRUSTED and record.success_count >= thresholds.habitual_success_count:
            if correction_rate <= thresholds.correction_rate_threshold:
                return LifecycleState.HABITUAL
        if current == LifecycleState.HABITUAL and record.opt_in_automated:
            if correction_rate <= thresholds.correction_rate_threshold:
                return LifecycleState.AUTOMATED
    return current


def record_event(
    signature: IntentSignature,
    event: LifecycleEvent,
    *,
    thresholds: LifecycleThresholds | None = None,
) -> LifecycleRecord:
    thresholds = thresholds or LifecycleThresholds()
    key = signature_key(signature)
    record = get_record(key)
    now = _now()
    if record is None:
        record = LifecycleRecord(
            signature_key=key,
            intent_name=signature.intent_name,
            category=signature.category,
            state=LifecycleState.DISCOVERED,
            first_seen_at=now,
            last_seen_at=now,
            usage_count=0,
            success_count=0,
            correction_count=0,
            last_mode_used=None,
            last_outcome=None,
            trust_score=None,
            opt_in_automated=False,
        )
    record.last_seen_at = now
    record.usage_count += 1
    if event.event_type == LifecycleEventType.EXECUTION_SUCCEEDED:
        record.success_count += 1
        record.last_outcome = "success"
    if event.event_type == LifecycleEventType.EXECUTION_FAILED:
        record.last_outcome = "fail"
    if event.event_type == LifecycleEventType.USER_ABORTED:
        record.last_outcome = "aborted"
    if event.event_type == LifecycleEventType.USER_CORRECTED_PLAN:
        record.correction_count += 1
    if event.mode_used:
        record.last_mode_used = event.mode_used.value
    new_state = next_state(record.state, event, record, thresholds)
    if new_state != record.state:
        _emit_state_change(record, new_state, event)
        record.state = new_state
    upsert_record(record)
    return record


def describe_lifecycle(record: LifecycleRecord) -> dict[str, Any]:
    hint = lifecycle_hint(record.state, record.category)
    return {
        "signature_key": record.signature_key,
        "intent": record.intent_name,
        "category": record.category.value,
        "state": record.state.value,
        "usage_count": record.usage_count,
        "success_count": record.success_count,
        "correction_count": record.correction_count,
        "last_seen_at": record.last_seen_at,
        "preferred_mode": hint.preferred_mode.value,
        "autonomy_cap": hint.autonomy_cap,
        "note": _state_note(record.state),
    }


def get_record(signature_key_value: str) -> LifecycleRecord | None:
    db_path = resolve_nervous_system_db_path()
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT signature_key, intent_name, category, state, first_seen_at, last_seen_at,
                       usage_count, success_count, correction_count, last_mode_used, last_outcome,
                       trust_score, opt_in_automated
                FROM intent_lifecycle
                WHERE signature_key = ?
                """,
                (signature_key_value,),
            ).fetchone()
    except sqlite3.OperationalError:
        return None
    if not row:
        return None
    return LifecycleRecord(
        signature_key=row[0],
        intent_name=row[1],
        category=IntentCategory(row[2]),
        state=LifecycleState(row[3]),
        first_seen_at=row[4],
        last_seen_at=row[5],
        usage_count=int(row[6]),
        success_count=int(row[7]),
        correction_count=int(row[8]),
        last_mode_used=row[9],
        last_outcome=row[10],
        trust_score=row[11],
        opt_in_automated=bool(row[12]),
    )


def upsert_record(record: LifecycleRecord) -> None:
    db_path = resolve_nervous_system_db_path()
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO intent_lifecycle
                  (signature_key, intent_name, category, state, first_seen_at, last_seen_at,
                   usage_count, success_count, correction_count, last_mode_used, last_outcome,
                   trust_score, opt_in_automated)
                VALUES
                  (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(signature_key) DO UPDATE SET
                  intent_name=excluded.intent_name,
                  category=excluded.category,
                  state=excluded.state,
                  last_seen_at=excluded.last_seen_at,
                  usage_count=excluded.usage_count,
                  success_count=excluded.success_count,
                  correction_count=excluded.correction_count,
                  last_mode_used=excluded.last_mode_used,
                  last_outcome=excluded.last_outcome,
                  trust_score=excluded.trust_score,
                  opt_in_automated=excluded.opt_in_automated
                """,
                (
                    record.signature_key,
                    record.intent_name,
                    record.category.value,
                    record.state.value,
                    record.first_seen_at,
                    record.last_seen_at,
                    record.usage_count,
                    record.success_count,
                    record.correction_count,
                    record.last_mode_used,
                    record.last_outcome,
                    record.trust_score,
                    1 if record.opt_in_automated else 0,
                ),
            )
            conn.commit()
    except sqlite3.OperationalError:
        return


def _state_note(state: LifecycleState) -> str:
    if state == LifecycleState.DISCOVERED:
        return "Newly seen intent; needs validation."
    if state == LifecycleState.BOOTSTRAPPED:
        return "Known intent; guidance recommended."
    if state == LifecycleState.GUIDED:
        return "User guided intent at least once."
    if state == LifecycleState.TRUSTED:
        return "Consistent success with low corrections."
    if state == LifecycleState.HABITUAL:
        return "Frequently successful; minimal confirmation."
    return "Eligible for automation under policy."


def _correction_rate(record: LifecycleRecord) -> float:
    if record.usage_count <= 0:
        return 0.0
    return record.correction_count / record.usage_count


def _stable_json(value: dict[str, Any]) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit_state_change(
    record: LifecycleRecord,
    new_state: LifecycleState,
    event: LifecycleEvent,
) -> None:
    payload = {
        "signature_key": record.signature_key,
        "from_state": record.state.value,
        "to_state": new_state.value,
        "reason": event.event_type.value,
        "timestamp": _now(),
    }
    logger.info("lifecycle.state_changed %s", payload)
