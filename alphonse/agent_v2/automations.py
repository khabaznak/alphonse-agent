"""Local worker events and event-triggered automations for Alphonse v2."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from alphonse.agent_v2.database import connect_database, default_database_path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict): return dict(value)
    try: parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError): return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


@dataclass(frozen=True)
class EventEnvelope:
    """Immutable scope resolved from trusted check-run state, never event payload."""

    publication_id: str
    worker_id: str
    actor_user_id: str
    owner_user_id: str
    project_id: str
    policy_id: str
    policy_revision: int
    check_run_id: str
    resource_id: str
    event_type: str
    event_version: str
    occurred_at: str
    observed_at: str
    connection_id: str = ""
    grant_id: str = ""
    grant_revision: int = 0
    schema_version: int = 1

    def valid(self) -> bool:
        required = (self.publication_id, self.worker_id, self.actor_user_id,
                    self.owner_user_id, self.project_id, self.policy_id,
                    self.check_run_id, self.resource_id, self.event_type, self.event_version,
                    self.occurred_at, self.observed_at)
        try:
            times_valid = all(datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is not None
                              for value in (self.occurred_at, self.observed_at))
        except (TypeError, ValueError, AttributeError):
            times_valid = False
        return (isinstance(self.connection_id, str) and isinstance(self.grant_id, str)
                and type(self.grant_revision) is int
                and times_valid and type(self.schema_version) is int and self.schema_version == 1 and all(isinstance(v, str) and v.strip() for v in required)
                and type(self.policy_revision) is int and self.policy_revision > 0
                and ((not self.connection_id and not self.grant_id and self.grant_revision == 0)
                     or (bool(self.connection_id and self.grant_id)
                         and type(self.grant_revision) is int and self.grant_revision > 0)))


class ManagedEventAuthority(Protocol):
    """Trusted service seam; implementations must read current authoritative state.

    resolve binds the registered adapter to its run, actor, account/resource,
    grant and policy revision. authorize checks active actors, worker, account,
    grant (including resource/operation/audience), project, policy and run at
    EACH call. Missing services or unknown versions must deny. No payload is an
    input to either authority decision. These are deterministic service checks.
    """

    def resolve(self, worker_id: str, check_run_id: str) -> EventEnvelope: ...
    def authorize(self, envelope: EventEnvelope, *, actor_user_id: str,
                  project_id: str, phase: str) -> bool: ...


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
        self._authorities: dict[str, ManagedEventAuthority] = {}
        self._memory: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "EventAutomationStore":
        return cls(default_database_path())

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

    def bind_managed_adapter(self, worker_id: str, authority: ManagedEventAuthority) -> None:
        """Daemon-only registration. Persist the fence even when services restart."""
        worker = self.get_worker(worker_id)
        if worker is None:
            raise ValueError("worker_not_registered")
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO v2_managed_event_workers(worker_id) VALUES(?)", (worker_id,))
            for kind in worker.allowed_event_types:
                conn.execute("INSERT OR IGNORE INTO v2_private_event_types(event_type) VALUES(?)", (kind,))
        self._authorities[worker_id] = authority

    def is_managed_worker(self, worker_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute("SELECT 1 FROM v2_managed_event_workers WHERE worker_id=?", (worker_id,)).fetchone() is not None

    def is_private_type(self, event_type: str) -> bool:
        with self._connect() as conn:
            return conn.execute("SELECT 1 FROM v2_private_event_types WHERE event_type=?", (event_type,)).fetchone() is not None

    def _authorized(self, envelope: EventEnvelope, actor: str, project: str, phase: str) -> bool:
        worker = self.get_worker(envelope.worker_id)
        authority = self._authorities.get(envelope.worker_id)
        if not envelope.valid() or not worker or not worker.enabled or not authority or not actor:
            return False
        definition = self.get_event_type(envelope.event_type, envelope.event_version)
        if (envelope.event_type not in worker.allowed_event_types or definition is None
                or not definition.enabled):
            return False
        # No implicit project-sharing or administrator bypass. V1 subscriptions
        # run as the original actor in the exact authorized project.
        if actor != envelope.actor_user_id or project != envelope.project_id:
            return False
        try:
            return authority.authorize(envelope, actor_user_id=actor, project_id=project, phase=phase) is True
        except Exception:
            return False

    def _subscription_authorized(self, automation: AutomationRecord | None, envelope: EventEnvelope, phase: str) -> bool:
        if automation is None:
            return False
        scope = automation.trigger.get("source_scope")
        if not isinstance(scope, dict):
            return False
        fields = ("worker_id", "actor_user_id", "owner_user_id", "project_id", "policy_id",
                  "policy_revision", "resource_id", "connection_id", "grant_id", "grant_revision")
        return (all(scope.get(key) == getattr(envelope, key) for key in fields)
                and self._authorized(envelope, automation.owner_user_id, automation.project_id, phase))

    def publish_managed(self, *, worker_id: str, check_run_id: str, event_id: str,
                        event_type: str, event_version: str, occurred_at: str,
                        payload: dict[str, Any]) -> dict[str, Any]:
        authority = self._authorities.get(worker_id)
        if not self.is_managed_worker(worker_id) or not self.is_private_type(event_type) or authority is None:
            return {"accepted": False, "reason": "event_scope_denied"}
        try:
            envelope = authority.resolve(worker_id, check_run_id)
            if (not isinstance(envelope, EventEnvelope) or envelope.worker_id != worker_id
                    or envelope.check_run_id != check_run_id
                    or envelope.event_type != event_type or envelope.event_version != event_version
                    or not self._authorized(envelope, envelope.actor_user_id, envelope.project_id, "publication")):
                raise ValueError("event_scope_denied")
        except Exception:
            return {"accepted": False, "reason": "event_scope_denied"}
        return self._publish(worker_id=worker_id, event_id=event_id, event_type=event_type,
                             event_version=event_version, occurred_at=envelope.occurred_at,
                             payload=payload, envelope=envelope)

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

    def create_event_automation(self, *, owner_user_id: str, name: str, prompt: str, event_type: str, event_version: str, filters: dict[str, Any] | None = None, project_id: str = "", origin_channel: dict[str, Any] | None = None, enabled: bool = True, source_scope: EventEnvelope | None = None) -> AutomationRecord:
        owner, title, text = str(owner_user_id or "").strip(), str(name or "").strip(), str(prompt or "").strip()
        if not owner or not title or not text: raise ValueError("automation_owner_name_prompt_required")
        project = str(project_id or "").strip()
        if not project: raise ValueError("automation_project_required")
        if self.get_event_type(event_type, event_version) is None: raise ValueError("event_type_not_registered")
        if self.is_private_type(event_type):
            if source_scope is None or not self._authorized(source_scope, owner, project, "subscription"):
                raise PermissionError("event_scope_denied")
            if source_scope.event_type != event_type or source_scope.event_version != event_version:
                raise PermissionError("event_scope_denied")
        elif source_scope is not None:
            raise ValueError("event_type_not_private")
        now = _now(); record = AutomationRecord(f"automation_{uuid4().hex[:16]}", owner, project, title, text, "event", {"event_type": str(event_type), "event_version": str(event_version), "filters": dict(filters or {}), **({"source_scope": asdict(source_scope)} if source_scope else {})}, dict(origin_channel or {}), "active" if enabled else "paused", now, now)
        with self._connect() as conn: conn.execute("INSERT INTO v2_automations(automation_id,owner_user_id,project_id,name,prompt,trigger_kind,trigger_json,origin_channel_json,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)", _automation_values(record))
        return record

    def list_automations(self, *, owner_user_id: str = "") -> list[AutomationRecord]:
        query, values = "SELECT * FROM v2_automations", []
        if owner_user_id: query += " WHERE owner_user_id=?"; values.append(str(owner_user_id))
        query += " ORDER BY updated_at DESC"
        with self._connect() as conn: rows = conn.execute(query, tuple(values)).fetchall()
        return [item for row in rows if (item := _automation(row))]

    def visible_automations(self, *, actor_user_id: str) -> list[AutomationRecord]:
        if not actor_user_id:
            return []
        visible = []
        for item in self.list_automations(owner_user_id=actor_user_id):
            if self.is_private_type(str(item.trigger.get("event_type") or "")):
                scope = _envelope(json.dumps(item.trigger.get("source_scope")))
                if scope is None or not self._subscription_authorized(item, scope, "catalog"):
                    continue
            visible.append(item)
        return visible

    def set_automation_enabled(self, automation_id: str, enabled: bool) -> AutomationRecord:
        with self._connect() as conn: conn.execute("UPDATE v2_automations SET status=?,updated_at=? WHERE automation_id=?", ("active" if enabled else "paused", _now(), str(automation_id)))
        item = self.get_automation(automation_id)
        if item is None: raise KeyError("automation_not_found")
        return item

    def get_automation(self, automation_id: str) -> AutomationRecord | None:
        with self._connect() as conn: row = conn.execute("SELECT * FROM v2_automations WHERE automation_id=?", (str(automation_id),)).fetchone()
        return _automation(row)

    def publish(self, *, worker_id: str, event_id: str, event_type: str, event_version: str, occurred_at: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.is_managed_worker(worker_id) or self.is_private_type(event_type):
            return {"accepted": False, "reason": "managed_adapter_required"}
        return self._publish(worker_id=worker_id, event_id=event_id, event_type=event_type,
                             event_version=event_version, occurred_at=occurred_at, payload=payload)

    def _publish(self, *, worker_id: str, event_id: str, event_type: str,
                 event_version: str, occurred_at: str, payload: dict[str, Any],
                 envelope: EventEnvelope | None = None) -> dict[str, Any]:
        worker = self.get_worker(worker_id); definition = self.get_event_type(event_type, event_version)
        if worker is None or not worker.enabled or str(event_type) not in worker.allowed_event_types: return {"accepted": False, "reason": "worker_not_authorized"}
        if definition is None or not definition.enabled: return {"accepted": False, "reason": "event_type_not_registered"}
        try: Draft202012Validator(definition.schema).validate(payload)
        except ValidationError: return {"accepted": False, "reason": "event_payload_invalid"}
        source_id = str(event_id or "").strip()
        if not source_id: return {"accepted": False, "reason": "event_id_required"}
        now = _now(); event_key = f"event_{uuid4().hex}"
        with self._connect() as conn:
            duplicate = conn.execute("SELECT event_key,envelope_json FROM v2_event_history WHERE worker_id=? AND source_event_id=?", (worker.worker_id, source_id)).fetchone()
            if duplicate and envelope is not None and _obj(duplicate["envelope_json"]) != asdict(envelope):
                return {"accepted": False, "reason": "event_scope_denied"}
            if duplicate: return {"accepted": True, "duplicate": True, "event_key": str(duplicate[0]), "matches": []}
            conn.execute("INSERT INTO v2_event_history(event_key,worker_id,source_event_id,event_type,event_version,occurred_at,payload_json,dispatch_count,created_at) VALUES(?,?,?,?,?,?,?,?,?)", (event_key, worker.worker_id, source_id, definition.event_type, definition.version, str(occurred_at or now), json.dumps(payload, sort_keys=True), 0, now))
            if envelope is not None:
                conn.execute("UPDATE v2_event_history SET envelope_json=? WHERE event_key=?", (json.dumps(asdict(envelope), sort_keys=True), event_key))
            matches = [automation for automation in self.list_automations() if automation.status == "active"
                       and ((envelope is None and "source_scope" not in automation.trigger)
                            or (envelope is not None and self._subscription_authorized(automation, envelope, "subscription")))
                       and _matches(automation, definition.event_type, definition.version, payload)]
            for automation in matches:
                conn.execute("INSERT INTO v2_automation_executions(execution_id,automation_id,event_key,status,created_at,updated_at) VALUES(?,?,?,?,?,?)", (f"automation_execution_{uuid4().hex[:16]}", automation.automation_id, event_key, "pending", now, now))
            conn.execute("UPDATE v2_event_history SET dispatch_count=? WHERE event_key=?", (len(matches), event_key))
            self._prune_history(conn, definition)
        return {"accepted": True, "duplicate": False, "event_key": event_key, "matches": [item.automation_id for item in matches]}

    def claim_event_executions(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT x.execution_id,x.automation_id,x.event_key,a.owner_user_id,a.project_id,a.prompt,a.origin_channel_json,e.worker_id,e.source_event_id,e.event_type,e.event_version,e.occurred_at,e.payload_json,e.envelope_json,a.status AS automation_status FROM v2_automation_executions x JOIN v2_automations a ON a.automation_id=x.automation_id JOIN v2_event_history e ON e.event_key=x.event_key WHERE x.status='pending' ORDER BY x.created_at LIMIT ?", (max(1, min(int(limit), 1000)),)).fetchall()
            eligible = []
            for row in rows:
                envelope = _envelope(row["envelope_json"])
                worker = self.get_worker(str(row["worker_id"]))
                allowed = (row["automation_status"] == "active" and worker is not None and worker.enabled
                           and (not self.is_private_type(str(row["event_type"])) or envelope is not None)
                           and (not row["envelope_json"] or (envelope is not None and self._subscription_authorized(self.get_automation(str(row["automation_id"])), envelope, "dispatch"))))
                if allowed:
                    eligible.append(row)
                else:
                    conn.execute("UPDATE v2_automation_executions SET status='denied',updated_at=? WHERE execution_id=?", (_now(), row["execution_id"]))
            for row in eligible: conn.execute("UPDATE v2_automation_executions SET status='claimed',updated_at=? WHERE execution_id=?", (_now(), str(row["execution_id"])))
        return [dict(row) for row in eligible]

    def mark_execution_enqueued(self, execution_id: str, message_id: str) -> None:
        with self._connect() as conn: conn.execute("UPDATE v2_automation_executions SET status='enqueued',queued_message_id=?,updated_at=? WHERE execution_id=?", (str(message_id), _now(), str(execution_id)))

    def expand_event(self, event_key: str, *, actor_user_id: str, project_id: str) -> dict[str, Any] | None:
        """Recheck current authority on every source expansion, including queued refs."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_event_history WHERE event_key=?", (event_key,)).fetchone()
        if row is None:
            return None
        envelope = _envelope(row["envelope_json"])
        if (self.is_private_type(str(row["event_type"])) or self.is_managed_worker(str(row["worker_id"]))) and not row["envelope_json"]:
            return None
        if row["envelope_json"] and (envelope is None or not self._authorized(envelope, actor_user_id, project_id, "expansion")):
            return None
        return {**dict(row), "payload": _obj(row["payload_json"]),
                "envelope": asdict(envelope) if envelope else None}

    def expand_execution_event(self, execution_id: str, event_key: str, *,
                               message_id: str, actor_user_id: str,
                               project_id: str) -> dict[str, Any] | None:
        """Resolve a queued reference against the server-owned subscription/run."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_automation_executions WHERE execution_id=?",
                               (execution_id,)).fetchone()
        if (row is None or row["event_key"] != event_key or row["status"] != "enqueued"
                or row["queued_message_id"] != message_id):
            return None
        automation = self.get_automation(str(row["automation_id"]))
        if (automation is None or automation.status != "active"
                or automation.owner_user_id != actor_user_id or automation.project_id != project_id):
            return None
        event = self.expand_event(event_key, actor_user_id=actor_user_id, project_id=project_id)
        if event is None:
            return None
        envelope = _envelope(event.get("envelope_json"))
        if envelope is not None and not self._subscription_authorized(automation, envelope, "expansion"):
            return None
        worker = self.get_worker(str(event["worker_id"]))
        definition = self.get_event_type(str(event["event_type"]), str(event["event_version"]))
        if (not worker or not worker.enabled or not definition or not definition.enabled
                or event["event_type"] not in worker.allowed_event_types):
            return None
        return event

    def read_event(self, event_key: str, *, actor_user_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT envelope_json FROM v2_event_history WHERE event_key=?", (event_key,)).fetchone()
        envelope = _envelope(row["envelope_json"]) if row else None
        return self.expand_event(event_key, actor_user_id=actor_user_id,
                                 project_id=envelope.project_id if envelope else "")

    def list_events(self, *, limit: int = 100, actor_user_id: str = "") -> list[dict[str, Any]]:
        # Filter before limiting: another owner's rows must not hide a reader's history.
        with self._connect() as conn:
            rows = conn.execute("SELECT event_key,envelope_json FROM v2_event_history ORDER BY created_at DESC").fetchall()
        result = []
        for row in rows:
            envelope = _envelope(row["envelope_json"])
            item = self.expand_event(str(row["event_key"]), actor_user_id=actor_user_id,
                                     project_id=envelope.project_id if envelope else "")
            if item is not None:
                result.append(item)
                if len(result) >= max(1, min(int(limit), 1000)):
                    break
        return result

    def _prune_history(self, conn: sqlite3.Connection, definition: EventTypeRecord) -> None:
        conn.execute("DELETE FROM v2_event_history WHERE event_key IN (SELECT event_key FROM v2_event_history WHERE event_type=? AND event_version=? ORDER BY created_at DESC LIMIT -1 OFFSET ?)", (definition.event_type, definition.version, definition.max_history))

    def _connect(self):
        if self._memory is not None: return _Connection(self._memory)
        return connect_database(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS v2_managed_event_workers(worker_id TEXT PRIMARY KEY) STRICT;
            CREATE TABLE IF NOT EXISTS v2_private_event_types(event_type TEXT PRIMARY KEY) STRICT;
            CREATE TABLE IF NOT EXISTS v2_event_workers(worker_id TEXT PRIMARY KEY,display_name TEXT NOT NULL,allowed_event_types_json TEXT NOT NULL,enabled INTEGER NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL) STRICT;
            CREATE TABLE IF NOT EXISTS v2_event_types(event_type TEXT NOT NULL,version TEXT NOT NULL,schema_json TEXT NOT NULL,max_history INTEGER NOT NULL,enabled INTEGER NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,PRIMARY KEY(event_type,version)) STRICT;
            CREATE TABLE IF NOT EXISTS v2_event_history(event_key TEXT PRIMARY KEY,worker_id TEXT NOT NULL,source_event_id TEXT NOT NULL,event_type TEXT NOT NULL,event_version TEXT NOT NULL,occurred_at TEXT NOT NULL,payload_json TEXT NOT NULL,dispatch_count INTEGER NOT NULL DEFAULT 0,created_at TEXT NOT NULL,UNIQUE(worker_id,source_event_id)) STRICT;
            CREATE TABLE IF NOT EXISTS v2_automations(automation_id TEXT PRIMARY KEY,owner_user_id TEXT NOT NULL,project_id TEXT NOT NULL DEFAULT '',name TEXT NOT NULL,prompt TEXT NOT NULL,trigger_kind TEXT NOT NULL,trigger_json TEXT NOT NULL,origin_channel_json TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL) STRICT;
            CREATE TABLE IF NOT EXISTS v2_automation_executions(execution_id TEXT PRIMARY KEY,automation_id TEXT NOT NULL,event_key TEXT NOT NULL,status TEXT NOT NULL,queued_message_id TEXT NOT NULL DEFAULT '',created_at TEXT NOT NULL,updated_at TEXT NOT NULL,UNIQUE(automation_id,event_key)) STRICT;
            """)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(v2_event_history)")}
            if "envelope_json" not in columns:
                conn.execute("ALTER TABLE v2_event_history ADD COLUMN envelope_json TEXT NOT NULL DEFAULT ''")


def _envelope(value: str) -> EventEnvelope | None:
    if not value:
        return None
    try:
        envelope = EventEnvelope(**json.loads(value))
        return envelope if envelope.valid() else None
    except (TypeError, ValueError):
        return None


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
