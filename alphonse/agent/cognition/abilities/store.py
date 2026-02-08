from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


class AbilitySpecStore:
    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or str(resolve_nervous_system_db_path())

    def list_enabled_specs(self) -> list[dict[str, Any]]:
        items = self.list_specs(enabled_only=True)
        specs: list[dict[str, Any]] = []
        for item in items:
            spec = item.get("spec")
            if isinstance(spec, dict):
                specs.append(spec)
        return specs

    def list_specs(self, *, enabled_only: bool = False, limit: int = 100) -> list[dict[str, Any]]:
        if not self.is_available():
            return []
        where = "WHERE enabled = 1" if enabled_only else ""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT intent_name, kind, tools_json, spec_json, enabled, source, created_at, updated_at
                FROM ability_specs
                {where}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [_row_to_item(row) for row in rows]

    def get_spec(self, intent_name: str) -> dict[str, Any] | None:
        if not self.is_available():
            return None
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT intent_name, kind, tools_json, spec_json, enabled, source, created_at, updated_at
                FROM ability_specs
                WHERE intent_name = ?
                """,
                (intent_name,),
            ).fetchone()
        return _row_to_item(row) if row else None

    def upsert_spec(
        self,
        intent_name: str,
        spec: dict[str, Any],
        *,
        enabled: bool = True,
        source: str = "user",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        tools = spec.get("tools") if isinstance(spec.get("tools"), list) else []
        kind = str(spec.get("kind") or "")
        payload = dict(spec)
        payload["intent_name"] = intent_name
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO ability_specs (
                  intent_name, kind, tools_json, spec_json, enabled, source, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(intent_name) DO UPDATE SET
                  kind = excluded.kind,
                  tools_json = excluded.tools_json,
                  spec_json = excluded.spec_json,
                  enabled = excluded.enabled,
                  source = excluded.source,
                  updated_at = excluded.updated_at
                """,
                (
                    intent_name,
                    kind,
                    json.dumps(tools),
                    json.dumps(payload),
                    1 if enabled else 0,
                    source,
                    now,
                    now,
                ),
            )

    def delete_spec(self, intent_name: str) -> bool:
        if not self.is_available():
            return False
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute("DELETE FROM ability_specs WHERE intent_name = ?", (intent_name,))
        return cur.rowcount > 0

    def is_available(self) -> bool:
        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='ability_specs'"
                ).fetchone()
                if not row:
                    return False
                columns = {col[1] for col in conn.execute("PRAGMA table_info(ability_specs)")}
        except sqlite3.Error:
            return False
        required = {
            "intent_name",
            "kind",
            "tools_json",
            "spec_json",
            "enabled",
            "source",
            "created_at",
            "updated_at",
        }
        return required.issubset(columns)


def _row_to_item(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    spec: dict[str, Any] = {}
    tools: list[str] = []
    try:
        parsed_spec = json.loads(str(row["spec_json"]))
        if isinstance(parsed_spec, dict):
            spec = parsed_spec
    except json.JSONDecodeError:
        spec = {}
    try:
        parsed_tools = json.loads(str(row["tools_json"]))
        if isinstance(parsed_tools, list):
            tools = [str(item) for item in parsed_tools]
    except json.JSONDecodeError:
        tools = []
    spec.setdefault("intent_name", str(row["intent_name"]))
    spec.setdefault("kind", str(row["kind"]))
    if not isinstance(spec.get("tools"), list):
        spec["tools"] = tools
    return {
        "intent_name": str(row["intent_name"]),
        "kind": str(row["kind"]),
        "tools": tools,
        "enabled": bool(row["enabled"]),
        "source": str(row["source"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "spec": spec,
    }
