"""Local worker events and event-triggered automations for Alphonse v2."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict): return dict(value)
    try: parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError): return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


@dataclass(frozen=True)
class WorkerRecord:
    worker_id: str; display_name: str; allowed_event_types: list[str]; enabled: bool; created_at: str; updated_at: str
    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(frozen=True)
class EventTypeRecord:
    event_type: str; version: str; schema: dict[str, Any]; max_history: int; enabled: bool; created_at: str; updated_at: str
    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(frozen=True)
class AutomationRecord:
    automation_id: str; owner_user_id: str; project_id: str; name: str; prompt: str; trigger_kind: str; trigger: dict[str, Any]; origin_channel: dict[str, Any]; status: str; created_at: str; updated_at: str
    def to_dict(self) -> dict[str, Any]: return asdict(self)


class EventAutomationStore:
    """Durable registry, event journal, and fan-out automation subscription store."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "EventAutomationStore":
        return cls(os.getenv("ALPHONSE_V2_AUTOMATIONS_DB_PATH") or str(Path.home() / ".alphonse" / "v2-automations.sqlite3"))

    def register_worker(self, *, worker_id: str, display_name: str, allowed_event_types: list[str], enabled: bool = True) -> WorkerRecord:
        worker = str(worker_id or "").strip(); name = str(display_name or "").strip()
        allowed = sorted({str(item).strip() for item in allowed_event_types if str(item).strip()})
        if not worker: raise ValueError("worker_id_required")
        if not name: raise ValueError("worker_display_name_required")
        now = _now()
        with self._connect() as conn:
            old = conn.execute("SELECT created_at FROM v2_event_workers WHERE worker_id=?", (worker,)).fetchone()
            conn.execute("INSERT INTO v2_event_workers(worker_id,display_name,allowed_event_types_json,enabled,created_at,updated_at) VALUES(?,?,?,?,?,?) ON CONFLICT(worker_id) DO UPDATE SET display_name=excluded.display_name,allowed_event_types_json=excluded.allowed_event_types_json,enabled=excluded.enabled,updated_at=excluded.updated_at", (worker, name, json.dumps(allowed), int(enabled), str(old[0]) if old else now, now))
        return self.get_worker(worker)  # type: ignore[return-value]

    def get_worker(self, worker_id: str) -> WorkerRecord | None:
        with self._connect() as conn: row = conn.execute("SELECT * FROM v2_event_workers WHERE worker_id=?", (str(worker_id),)).fetchone()
        return _worker(row)

    def list_workers(self) -> list[WorkerRecord]:
        with self._connect() as conn: rows = conn.execute("SELECT * FROM v2_event_workers ORDER BY display_name,worker_id").fetchall()
        return [item for row in rows if (item := _worker(row))]

    def register_event_type(self, *, event_type: str, version: str, schema: dict[str, Any], max_history: int = 500, enabled: bool = True) -> EventTypeRecord:
        kind, ver = str(event_type or "").strip(), str(version or "").strip()
        if not kind: raise ValueError("event_type_required")
        if not ver: raise ValueError("event_version_required")
        try: Draft202012Validator.check_schema(schema)
        except SchemaError as exc: raise ValueError(f"event_schema_invalid: {exc.message}") from exc
        now = _now()
        with self._connect() as conn:
            old = conn.execute("SELECT created_at FROM v2_event_types WHERE event_type=? AND version=?", (kind, ver)).fetchone()
            conn.execute("INSERT INTO v2_event_types(event_type,version,schema_json,max_history,enabled,created_at,updated_at) VALUES(?,?,?,?,?,?,?) ON CONFLICT(event_type,version) DO UPDATE SET schema_json=excluded.schema_json,max_history=excluded.max_history,enabled=excluded.enabled,updated_at=excluded.updated_at", (kind, ver, json.dumps(schema, sort_keys=True), max(1, int(max_history)), int(enabled), str(old[0]) if old else now, now))
        return self.get_event_type(kind, ver)  # type: ignore[return-value]

    def get_event_type(self, event_type: str, version: str) -> EventTypeRecord | None:
        with self._connect() as conn: row = conn.execute("SELECT * FROM v2_event_types WHERE event_type=? AND version=?", (str(event_type), str(version))).fetchone()
        return _event_type(row)

    def list_event_types(self) -> list[EventTypeRecord]:
        with self._connect() as conn: rows = conn.execute("SELECT * FROM v2_event_types ORDER BY event_type,version").fetchall()
        return [item for row in rows if (item := _event_type(row))]

    def create_event_automation(self, *, owner_user_id: str, name: str, prompt: str, event_type: str, event_version: str, filters: dict[str, Any] | None = None, project_id: str = "", origin_channel: dict[str, Any] | None = None, enabled: bool = True) -> AutomationRecord:
        owner, title, text = str(owner_user_id or "").strip(), str(name or "").strip(), str(prompt or "").strip()
        if not owner or not title or not text: raise ValueError("automation_owner_name_prompt_required")
        if self.get_event_type(event_type, event_version) is None: raise ValueError("event_type_not_registered")
        now = _now(); record = AutomationRecord(f"automation_{uuid4().hex[:16]}", owner, str(project_id or "").strip(), title, text, "event", {"event_type": str(event_type), "event_version": str(event_version), "filters": dict(filters or {})}, dict(origin_channel or {}), "active" if enabled else "paused", now, now)
        with self._connect() as conn: conn.execute("INSERT INTO v2_automations(automation_id,owner_user_id,project_id,name,prompt,trigger_kind,trigger_json,origin_channel_json,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)", _automation_values(record))
        return record

    def list_automations(self, *, owner_user_id: str = "") -> list[AutomationRecord]:
        query, values = "SELECT * FROM v2_automations", []
        if owner_user_id: query += " WHERE owner_user_id=?"; values.append(str(owner_user_id))
        query += " ORDER BY updated_at DESC"
        with self._connect() as conn: rows = conn.execute(query, tuple(values)).fetchall()
        return [item for row in rows if (item := _automation(row))]

    def set_automation_enabled(self, automation_id: str, enabled: bool) -> AutomationRecord:
        with self._connect() as conn: conn.execute("UPDATE v2_automations SET status=?,updated_at=? WHERE automation_id=?", ("active" if enabled else "paused", _now(), str(automation_id)))
        item = self.get_automation(automation_id)
        if item is None: raise KeyError("automation_not_found")
        return item

    def get_automation(self, automation_id: str) -> AutomationRecord | None:
        with self._connect() as conn: row = conn.execute("SELECT * FROM v2_automations WHERE automation_id=?", (str(automation_id),)).fetchone()
        return _automation(row)

    def publish(self, *, worker_id: str, event_id: str, event_type: str, event_version: str, occurred_at: str, payload: dict[str, Any]) -> dict[str, Any]:
        worker = self.get_worker(worker_id); definition = self.get_event_type(event_type, event_version)
        if worker is None or not worker.enabled or str(event_type) not in worker.allowed_event_types: return {"accepted": False, "reason": "worker_not_authorized"}
        if definition is None or not definition.enabled: return {"accepted": False, "reason": "event_type_not_registered"}
        try: Draft202012Validator(definition.schema).validate(payload)
        except ValidationError: return {"accepted": False, "reason": "event_payload_invalid"}
        source_id = str(event_id or "").strip()
        if not source_id: return {"accepted": False, "reason": "event_id_required"}
        now = _now(); event_key = f"event_{uuid4().hex}"
        with self._connect() as conn:
            duplicate = conn.execute("SELECT event_key FROM v2_event_history WHERE worker_id=? AND source_event_id=?", (worker.worker_id, source_id)).fetchone()
            if duplicate: return {"accepted": True, "duplicate": True, "event_key": str(duplicate[0]), "matches": []}
            conn.execute("INSERT INTO v2_event_history(event_key,worker_id,source_event_id,event_type,event_version,occurred_at,payload_json,dispatch_count,created_at) VALUES(?,?,?,?,?,?,?,?,?)", (event_key, worker.worker_id, source_id, definition.event_type, definition.version, str(occurred_at or now), json.dumps(payload, sort_keys=True), 0, now))
            matches = [automation for automation in self.list_automations() if automation.status == "active" and _matches(automation, definition.event_type, definition.version, payload)]
            for automation in matches:
                conn.execute("INSERT INTO v2_automation_executions(execution_id,automation_id,event_key,status,created_at,updated_at) VALUES(?,?,?,?,?,?)", (f"automation_execution_{uuid4().hex[:16]}", automation.automation_id, event_key, "pending", now, now))
            conn.execute("UPDATE v2_event_history SET dispatch_count=? WHERE event_key=?", (len(matches), event_key))
            self._prune_history(conn, definition)
        return {"accepted": True, "duplicate": False, "event_key": event_key, "matches": [item.automation_id for item in matches]}

    def claim_event_executions(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT x.execution_id,x.automation_id,x.event_key,a.owner_user_id,a.project_id,a.prompt,a.origin_channel_json,e.worker_id,e.source_event_id,e.event_type,e.event_version,e.occurred_at,e.payload_json FROM v2_automation_executions x JOIN v2_automations a ON a.automation_id=x.automation_id JOIN v2_event_history e ON e.event_key=x.event_key WHERE x.status='pending' ORDER BY x.created_at LIMIT ?", (max(1, min(int(limit), 1000)),)).fetchall()
            for row in rows: conn.execute("UPDATE v2_automation_executions SET status='claimed',updated_at=? WHERE execution_id=?", (_now(), str(row["execution_id"])))
        return [dict(row) for row in rows]

    def mark_execution_enqueued(self, execution_id: str, message_id: str) -> None:
        with self._connect() as conn: conn.execute("UPDATE v2_automation_executions SET status='enqueued',queued_message_id=?,updated_at=? WHERE execution_id=?", (str(message_id), _now(), str(execution_id)))

    def list_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn: rows = conn.execute("SELECT * FROM v2_event_history ORDER BY created_at DESC LIMIT ?", (max(1, min(int(limit), 1000)),)).fetchall()
        return [{**dict(row), "payload": _obj(row["payload_json"])} for row in rows]

    def _prune_history(self, conn: sqlite3.Connection, definition: EventTypeRecord) -> None:
        conn.execute("DELETE FROM v2_event_history WHERE event_key IN (SELECT event_key FROM v2_event_history WHERE event_type=? AND event_version=? ORDER BY created_at DESC LIMIT -1 OFFSET ?)", (definition.event_type, definition.version, definition.max_history))

    def _connect(self):
        if self._memory is not None: return _Connection(self._memory)
        path = Path(self.db_path); path.parent.mkdir(parents=True, exist_ok=True); conn = sqlite3.connect(path); conn.row_factory = sqlite3.Row; return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn: conn.executescript("""
            CREATE TABLE IF NOT EXISTS v2_event_workers(worker_id TEXT PRIMARY KEY,display_name TEXT NOT NULL,allowed_event_types_json TEXT NOT NULL,enabled INTEGER NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL) STRICT;
            CREATE TABLE IF NOT EXISTS v2_event_types(event_type TEXT NOT NULL,version TEXT NOT NULL,schema_json TEXT NOT NULL,max_history INTEGER NOT NULL,enabled INTEGER NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,PRIMARY KEY(event_type,version)) STRICT;
            CREATE TABLE IF NOT EXISTS v2_event_history(event_key TEXT PRIMARY KEY,worker_id TEXT NOT NULL,source_event_id TEXT NOT NULL,event_type TEXT NOT NULL,event_version TEXT NOT NULL,occurred_at TEXT NOT NULL,payload_json TEXT NOT NULL,dispatch_count INTEGER NOT NULL DEFAULT 0,created_at TEXT NOT NULL,UNIQUE(worker_id,source_event_id)) STRICT;
            CREATE TABLE IF NOT EXISTS v2_automations(automation_id TEXT PRIMARY KEY,owner_user_id TEXT NOT NULL,project_id TEXT NOT NULL DEFAULT '',name TEXT NOT NULL,prompt TEXT NOT NULL,trigger_kind TEXT NOT NULL,trigger_json TEXT NOT NULL,origin_channel_json TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL) STRICT;
            CREATE TABLE IF NOT EXISTS v2_automation_executions(execution_id TEXT PRIMARY KEY,automation_id TEXT NOT NULL,event_key TEXT NOT NULL,status TEXT NOT NULL,queued_message_id TEXT NOT NULL DEFAULT '',created_at TEXT NOT NULL,updated_at TEXT NOT NULL,UNIQUE(automation_id,event_key)) STRICT;
        """)


class _Connection:
    def __init__(self, conn): self.conn = conn
    def __enter__(self): return self.conn
    def __exit__(self, typ, value, tb): self.conn.commit() if typ is None else self.conn.rollback(); return False


def _worker(row): return WorkerRecord(str(row["worker_id"]), str(row["display_name"]), list(json.loads(row["allowed_event_types_json"])), bool(row["enabled"]), str(row["created_at"]), str(row["updated_at"])) if row else None
def _event_type(row): return EventTypeRecord(str(row["event_type"]), str(row["version"]), _obj(row["schema_json"]), int(row["max_history"]), bool(row["enabled"]), str(row["created_at"]), str(row["updated_at"])) if row else None
def _automation(row): return AutomationRecord(str(row["automation_id"]), str(row["owner_user_id"]), str(row["project_id"]), str(row["name"]), str(row["prompt"]), str(row["trigger_kind"]), _obj(row["trigger_json"]), _obj(row["origin_channel_json"]), str(row["status"]), str(row["created_at"]), str(row["updated_at"])) if row else None
def _automation_values(record): return (record.automation_id,record.owner_user_id,record.project_id,record.name,record.prompt,record.trigger_kind,json.dumps(record.trigger,sort_keys=True),json.dumps(record.origin_channel,sort_keys=True),record.status,record.created_at,record.updated_at)
def _matches(item, event_type, version, payload):
    trigger = item.trigger
    return item.trigger_kind == "event" and str(trigger.get("event_type")) == event_type and str(trigger.get("event_version")) == version and all(payload.get(key) == value for key,value in _obj(trigger.get("filters")).items())
