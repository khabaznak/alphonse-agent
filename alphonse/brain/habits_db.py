from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


@dataclass(frozen=True)
class Habit:
    habit_id: str
    name: str
    trigger: str
    conditions_json: dict[str, Any]
    plan_json: dict[str, Any]
    version: int
    enabled: bool
    created_at: str
    updated_at: str
    success_count: int
    fail_count: int
    last_success_at: str | None
    last_fail_at: str | None
    menu_snapshot_hash: str | None


@dataclass(frozen=True)
class PlanRun:
    run_id: str
    habit_id: str | None
    plan_id: str
    trigger: str
    correlation_id: str
    status: str
    resolution: str | None
    resolved_via: str | None
    started_at: str
    ended_at: str | None
    state_json: dict[str, Any]
    scheduled_json: list[dict[str, Any]] | None
    plan_json: dict[str, Any]


def _connect() -> sqlite3.Connection:
    path = resolve_nervous_system_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_audit(event_type: str, correlation_id: str | None, payload: dict[str, Any]) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO audit_log (id, event_type, correlation_id, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                event_type,
                correlation_id,
                json.dumps(payload, ensure_ascii=True),
                _utcnow(),
            ),
        )


def list_enabled_habits_for_trigger(trigger: str) -> list[Habit]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT habit_id, name, trigger, conditions_json, plan_json, version, enabled,
                   created_at, updated_at, success_count, fail_count, last_success_at,
                   last_fail_at, menu_snapshot_hash
            FROM habits
            WHERE trigger = ? AND enabled = 1
            """,
            (trigger,),
        ).fetchall()
    return [_row_to_habit(row) for row in rows]


def get_habit(habit_id: str) -> Habit | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT habit_id, name, trigger, conditions_json, plan_json, version, enabled,
                   created_at, updated_at, success_count, fail_count, last_success_at,
                   last_fail_at, menu_snapshot_hash
            FROM habits
            WHERE habit_id = ?
            """,
            (habit_id,),
        ).fetchone()
    if not row:
        return None
    return _row_to_habit(row)


def create_habit(
    *,
    name: str,
    trigger: str,
    conditions: dict[str, Any],
    plan: dict[str, Any],
    version: int = 1,
    enabled: bool = True,
    menu_snapshot_hash: str | None = None,
) -> Habit:
    habit_id = str(uuid4())
    now = _utcnow()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO habits (
              habit_id, name, trigger, conditions_json, plan_json, version, enabled,
              created_at, updated_at, success_count, fail_count, last_success_at,
              last_fail_at, menu_snapshot_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, NULL, NULL, ?)
            """,
            (
                habit_id,
                name,
                trigger,
                json.dumps(conditions, ensure_ascii=True),
                json.dumps(plan, ensure_ascii=True),
                version,
                1 if enabled else 0,
                now,
                now,
                menu_snapshot_hash,
            ),
        )
    return get_habit(habit_id)  # type: ignore[return-value]


def disable_habit(habit_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE habits SET enabled = 0, updated_at = ? WHERE habit_id = ?",
            (_utcnow(), habit_id),
        )


def create_plan_run(
    *,
    habit_id: str | None,
    plan_id: str,
    trigger: str,
    correlation_id: str,
    state: dict[str, Any],
    plan_json: dict[str, Any],
) -> PlanRun:
    now = _utcnow()
    run_id = str(uuid4())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO plan_runs (
              run_id, habit_id, plan_id, trigger, correlation_id, status, resolution,
              resolved_via, started_at, ended_at, state_json, scheduled_json, plan_json
            ) VALUES (?, ?, ?, ?, ?, 'running', NULL, NULL, ?, NULL, ?, NULL, ?)
            """,
            (
                run_id,
                habit_id,
                plan_id,
                trigger,
                correlation_id,
                now,
                json.dumps(state, ensure_ascii=True),
                json.dumps(plan_json, ensure_ascii=True),
            ),
        )
    return get_plan_run(run_id)  # type: ignore[return-value]


def get_plan_run(run_id: str) -> PlanRun | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT run_id, habit_id, plan_id, trigger, correlation_id, status, resolution,
                   resolved_via, started_at, ended_at, state_json, scheduled_json, plan_json
            FROM plan_runs WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
    if not row:
        return None
    return _row_to_plan_run(row)


def get_plan_run_by_correlation(correlation_id: str) -> PlanRun | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT run_id, habit_id, plan_id, trigger, correlation_id, status, resolution,
                   resolved_via, started_at, ended_at, state_json, scheduled_json, plan_json
            FROM plan_runs WHERE correlation_id = ?
            """,
            (correlation_id,),
        ).fetchone()
    if not row:
        return None
    return _row_to_plan_run(row)


def update_plan_run_status(
    run_id: str,
    *,
    status: str,
    resolution: str | None = None,
    resolved_via: str | None = None,
    ended_at: str | None = None,
    scheduled: list[dict[str, Any]] | None = None,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE plan_runs
            SET status = ?, resolution = ?, resolved_via = ?, ended_at = ?, scheduled_json = ?
            WHERE run_id = ?
            """,
            (
                status,
                resolution,
                resolved_via,
                ended_at,
                json.dumps(scheduled, ensure_ascii=True) if scheduled is not None else None,
                run_id,
            ),
        )


def insert_delivery_receipt(
    *,
    run_id: str | None,
    pairing_id: str | None,
    stage_id: str | None,
    action_id: str | None,
    skill: str | None,
    channel: str | None,
    status: str,
    details: dict[str, Any] | None = None,
) -> str:
    receipt_id = str(uuid4())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO delivery_receipts (
              receipt_id, run_id, pairing_id, stage_id, action_id, skill, channel, status,
              details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt_id,
                run_id,
                pairing_id,
                stage_id,
                action_id,
                skill,
                channel,
                status,
                json.dumps(details or {}, ensure_ascii=True),
                _utcnow(),
            ),
        )
    return receipt_id


def record_habit_outcome(habit_id: str, success: bool) -> None:
    now = _utcnow()
    if success:
        sql = """
        UPDATE habits
        SET success_count = success_count + 1,
            last_success_at = ?,
            updated_at = ?
        WHERE habit_id = ?
        """
        params = (now, now, habit_id)
    else:
        sql = """
        UPDATE habits
        SET fail_count = fail_count + 1,
            last_fail_at = ?,
            updated_at = ?
        WHERE habit_id = ?
        """
        params = (now, now, habit_id)
    with _connect() as conn:
        conn.execute(sql, params)


def export_habit(habit_id: str) -> dict[str, Any] | None:
    habit = get_habit(habit_id)
    if not habit:
        return None
    return {
        "habit_id": habit.habit_id,
        "name": habit.name,
        "trigger": habit.trigger,
        "conditions_json": habit.conditions_json,
        "plan_json": habit.plan_json,
        "version": habit.version,
        "enabled": habit.enabled,
    }


def import_habit(data: dict[str, Any]) -> Habit:
    name = str(data.get("name") or "Imported Habit")
    trigger = str(data.get("trigger") or "")
    conditions = dict(data.get("conditions_json") or {})
    plan = dict(data.get("plan_json") or {})
    version = int(data.get("version") or 1)
    return create_habit(
        name=name,
        trigger=trigger,
        conditions=conditions,
        plan=plan,
        version=version,
        enabled=bool(data.get("enabled", True)),
    )


def _row_to_habit(row: sqlite3.Row) -> Habit:
    return Habit(
        habit_id=row["habit_id"],
        name=row["name"],
        trigger=row["trigger"],
        conditions_json=json.loads(row["conditions_json"] or "{}"),
        plan_json=json.loads(row["plan_json"] or "{}"),
        version=row["version"],
        enabled=bool(row["enabled"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        success_count=row["success_count"],
        fail_count=row["fail_count"],
        last_success_at=row["last_success_at"],
        last_fail_at=row["last_fail_at"],
        menu_snapshot_hash=row["menu_snapshot_hash"],
    )


def _row_to_plan_run(row: sqlite3.Row) -> PlanRun:
    return PlanRun(
        run_id=row["run_id"],
        habit_id=row["habit_id"],
        plan_id=row["plan_id"],
        trigger=row["trigger"],
        correlation_id=row["correlation_id"],
        status=row["status"],
        resolution=row["resolution"],
        resolved_via=row["resolved_via"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        state_json=json.loads(row["state_json"] or "{}"),
        scheduled_json=json.loads(row["scheduled_json"] or "null"),
        plan_json=json.loads(row["plan_json"] or "{}"),
    )
