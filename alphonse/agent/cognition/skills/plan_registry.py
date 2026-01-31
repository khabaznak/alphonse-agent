from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


@dataclass(frozen=True)
class PlanSpec:
    plan_kind: str
    plan_version: int
    json_schema: dict
    example: str | None
    executor_key: str | None


def list_enabled_plan_specs() -> list[PlanSpec]:
    query = """
        SELECT pk.plan_kind, pkv.plan_version, pkv.json_schema, pkv.example, pe.executor_key
        FROM plan_kinds pk
        JOIN plan_kind_versions pkv ON pkv.plan_kind = pk.plan_kind
        LEFT JOIN plan_executors pe ON pe.plan_kind = pkv.plan_kind AND pe.plan_version = pkv.plan_version
        WHERE pk.is_enabled = 1 AND pkv.is_deprecated = 0
        ORDER BY pk.plan_kind, pkv.plan_version
    """
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        rows = conn.execute(query).fetchall()
    return [
        PlanSpec(
            plan_kind=row[0],
            plan_version=int(row[1]),
            json_schema=_parse_json(row[2]) or {},
            example=row[3],
            executor_key=row[4],
        )
        for row in rows
    ]


def get_plan_spec(plan_kind: str, plan_version: int) -> PlanSpec | None:
    query = """
        SELECT pk.plan_kind, pkv.plan_version, pkv.json_schema, pkv.example, pe.executor_key
        FROM plan_kinds pk
        JOIN plan_kind_versions pkv ON pkv.plan_kind = pk.plan_kind
        LEFT JOIN plan_executors pe ON pe.plan_kind = pkv.plan_kind AND pe.plan_version = pkv.plan_version
        WHERE pk.is_enabled = 1 AND pk.plan_kind = ? AND pkv.plan_version = ?
        LIMIT 1
    """
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(query, (plan_kind, plan_version)).fetchone()
    if not row:
        return None
    return PlanSpec(
        plan_kind=row[0],
        plan_version=int(row[1]),
        json_schema=_parse_json(row[2]) or {},
        example=row[3],
        executor_key=row[4],
    )


def _parse_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
