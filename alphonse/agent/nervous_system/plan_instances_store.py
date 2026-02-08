from __future__ import annotations

import json
import sqlite3
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def insert_plan_instance(plan: dict[str, Any], status: str) -> None:
    db_path = resolve_nervous_system_db_path()
    actor = plan.get("actor") if isinstance(plan.get("actor"), dict) else {}
    channel = actor.get("channel") if isinstance(actor.get("channel"), dict) else {}
    payload = plan.get("payload") if isinstance(plan.get("payload"), dict) else {}
    intent_evidence = (
        plan.get("intent_evidence") if isinstance(plan.get("intent_evidence"), dict) else {}
    )
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
                plan.get("plan_id"),
                plan.get("plan_kind"),
                int(plan.get("plan_version") or 1),
                plan.get("correlation_id"),
                status,
                actor.get("person_id"),
                channel.get("type"),
                channel.get("target"),
                float(plan.get("intent_confidence") or 0.0),
                json.dumps(payload),
                json.dumps(intent_evidence),
                plan.get("original_text"),
                plan.get("created_at"),
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
