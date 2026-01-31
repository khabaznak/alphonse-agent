from __future__ import annotations

import json
import sqlite3
from typing import Any

from alphonse.agent.cognition.skills.command_plans import CreateReminderPlan
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def insert_timed_signal_from_plan(plan: CreateReminderPlan) -> None:
    db_path = resolve_nervous_system_db_path()
    target_ref = plan.payload.target.person_ref
    target_id = target_ref.id or target_ref.name
    payload = {
        "plan": plan.model_dump(),
        "message": plan.payload.message.text,
        "person_id": target_id,
    }
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO timed_signals
              (id, trigger_at, next_trigger_at, rrule, timezone, status, fired_at, attempt_count,
               signal_type, payload, target, origin, correlation_id)
            VALUES
              (?, ?, ?, ?, ?, 'pending', NULL, 0, ?, ?, ?, ?, ?)
            """,
            (
                plan.plan_id,
                plan.payload.schedule.trigger_at,
                None,
                plan.payload.schedule.rrule,
                plan.payload.schedule.timezone,
                "reminder",
                json.dumps(payload),
                target_id,
                plan.source,
                plan.correlation_id,
            ),
        )
        conn.commit()
