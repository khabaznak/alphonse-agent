from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_observability_db_path

_MAX_DETAIL_CHARS = 4096
_MAX_ROWS_DEFAULT = 1_000_000
_NON_ERROR_TTL_DAYS_DEFAULT = 14
_ERROR_TTL_DAYS_DEFAULT = 30
_MAINTENANCE_INTERVAL_SECONDS_DEFAULT = 6 * 60 * 60

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trace_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  level TEXT NOT NULL,
  event TEXT NOT NULL,
  correlation_id TEXT,
  channel TEXT,
  user_id TEXT,
  node TEXT,
  cycle INTEGER,
  status TEXT,
  tool TEXT,
  error_code TEXT,
  latency_ms INTEGER,
  detail_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_trace_events_correlation_created
  ON trace_events (correlation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_trace_events_event_created
  ON trace_events (event, created_at);
CREATE INDEX IF NOT EXISTS idx_trace_events_level_created
  ON trace_events (level, created_at);
CREATE INDEX IF NOT EXISTS idx_trace_events_channel_created
  ON trace_events (channel, created_at);

CREATE TABLE IF NOT EXISTS trace_daily_rollups (
  day TEXT NOT NULL,
  event TEXT NOT NULL,
  level TEXT NOT NULL,
  count INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (day, event, level)
);
"""

_LOCK = threading.Lock()
_LAST_MAINTENANCE_AT = 0.0


def write_task_event(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        return
    now = datetime.now(timezone.utc)
    created_at = _to_iso(payload.get("ts"), fallback=now)
    detail_json = _truncate_json(payload)
    with _connect() as conn:
        _ensure_schema(conn)
        conn.execute(
            """
            INSERT INTO trace_events (
              created_at, level, event, correlation_id, channel, user_id, node,
              cycle, status, tool, error_code, latency_ms, detail_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                _as_text(payload.get("level"), "info"),
                _as_text(payload.get("event"), "unknown_event"),
                _as_nullable_text(payload.get("correlation_id")),
                _as_nullable_text(payload.get("channel")),
                _as_nullable_text(payload.get("user_id")),
                _as_nullable_text(payload.get("node")),
                _as_nullable_int(payload.get("cycle")),
                _as_nullable_text(payload.get("status")),
                _as_nullable_text(payload.get("tool")),
                _as_nullable_text(payload.get("error_code")),
                _as_nullable_int(payload.get("latency_ms")),
                detail_json,
            ),
        )
        day = created_at[:10]
        event = _as_text(payload.get("event"), "unknown_event")
        level = _as_text(payload.get("level"), "info")
        conn.execute(
            """
            INSERT INTO trace_daily_rollups(day, event, level, count)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(day, event, level)
            DO UPDATE SET count = count + 1
            """,
            (day, event, level),
        )
        conn.commit()
    _maybe_run_maintenance()


def run_maintenance(force: bool = False) -> None:
    with _LOCK:
        global _LAST_MAINTENANCE_AT
        now_monotonic = time.monotonic()
        interval = _env_int("ALPHONSE_OBSERVABILITY_MAINTENANCE_SECONDS", _MAINTENANCE_INTERVAL_SECONDS_DEFAULT)
        if not force and now_monotonic - _LAST_MAINTENANCE_AT < max(1, interval):
            return
        with _connect() as conn:
            _ensure_schema(conn)
            _prune_old_rows(conn)
            _enforce_max_rows(conn)
            conn.commit()
        _LAST_MAINTENANCE_AT = now_monotonic


def _maybe_run_maintenance() -> None:
    try:
        run_maintenance(force=False)
    except Exception:
        return


def _prune_old_rows(conn: sqlite3.Connection) -> None:
    now = datetime.now(timezone.utc)
    non_error_ttl = _env_int("ALPHONSE_OBSERVABILITY_NON_ERROR_TTL_DAYS", _NON_ERROR_TTL_DAYS_DEFAULT)
    error_ttl = _env_int("ALPHONSE_OBSERVABILITY_ERROR_TTL_DAYS", _ERROR_TTL_DAYS_DEFAULT)
    non_error_cutoff = (now - timedelta(days=max(1, non_error_ttl))).isoformat()
    error_cutoff = (now - timedelta(days=max(1, error_ttl))).isoformat()
    conn.execute(
        """
        DELETE FROM trace_events
        WHERE created_at < ?
          AND lower(coalesce(level, 'info')) NOT IN ('warning', 'error')
        """,
        (non_error_cutoff,),
    )
    conn.execute(
        """
        DELETE FROM trace_events
        WHERE created_at < ?
          AND lower(coalesce(level, 'info')) IN ('warning', 'error')
        """,
        (error_cutoff,),
    )


def _enforce_max_rows(conn: sqlite3.Connection) -> None:
    max_rows = _env_int("ALPHONSE_OBSERVABILITY_MAX_ROWS", _MAX_ROWS_DEFAULT)
    if max_rows < 1:
        return
    row = conn.execute("SELECT COUNT(*) FROM trace_events").fetchone()
    total = int(row[0]) if row else 0
    overflow = total - max_rows
    if overflow <= 0:
        return
    conn.execute(
        """
        DELETE FROM trace_events
        WHERE id IN (
          SELECT id FROM trace_events
          ORDER BY created_at ASC, id ASC
          LIMIT ?
        )
        """,
        (overflow,),
    )


def _connect() -> sqlite3.Connection:
    path = resolve_observability_db_path()
    return sqlite3.connect(path)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _to_iso(value: Any, fallback: datetime) -> str:
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()
        except ValueError:
            pass
    return fallback.isoformat()


def _as_text(value: Any, default: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def _as_nullable_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_nullable_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truncate_json(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
    if len(raw) <= _MAX_DETAIL_CHARS:
        return raw
    prefix_len = max(1, _MAX_DETAIL_CHARS - 64)
    compact = {
        "truncated": True,
        "prefix": raw[:prefix_len],
    }
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
