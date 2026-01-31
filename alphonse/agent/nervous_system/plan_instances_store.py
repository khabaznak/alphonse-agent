from __future__ import annotations

import json
import sqlite3

from alphonse.agent.cognition.skills.command_plans import CommandPlan
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def insert_plan_instance(plan: CommandPlan, status: str) -> None:
    db_path = resolve_nervous_system_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO plan_instances
              (plan_id, plan_kind, plan_version, correlation_id, status, actor_person_id,
               source_channel_type, source_channel_target, intent_confidence, payload,
               intent_evidence, original_text, created_at)
            VALUES
              (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan.plan_id,
                plan.plan_kind,
                plan.plan_version,
                plan.correlation_id,
                status,
                plan.actor.person_id,
                plan.actor.channel.type,
                plan.actor.channel.target,
                float(plan.intent_confidence),
                json.dumps(plan.payload.model_dump()),
                json.dumps(plan.intent_evidence.model_dump()),
                plan.original_text,
                plan.created_at,
            ),
        )
        conn.commit()


def update_plan_instance_status(plan_id: str, status: str) -> None:
    db_path = resolve_nervous_system_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE plan_instances SET status = ? WHERE plan_id = ?",
            (status, plan_id),
        )
        conn.commit()
