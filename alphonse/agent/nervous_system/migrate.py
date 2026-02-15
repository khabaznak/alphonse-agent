"""Apply schema to the nervous system SQLite database."""

from __future__ import annotations

import sqlite3
import sys
import os
from pathlib import Path


def default_db_path() -> Path:
    return Path(__file__).resolve().parent / "db" / "nerve-db"


def apply_schema(db_path: Path) -> None:
    schema_path = Path(__file__).resolve().parent / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_sql)
        _ensure_timed_signal_columns(conn)
        _ensure_timed_signal_statuses(conn)
        _ensure_paired_device_columns(conn)
        _ensure_pairing_columns(conn)
        _ensure_delivery_receipts_columns(conn)
        _ensure_intent_specs_columns(conn)
        _ensure_principals_constraints(conn)
        _ensure_users_table(conn)
        _ensure_prompt_template_columns(conn)
        _ensure_sandbox_directories(conn)


def main() -> None:
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    apply_schema(db_path)
    print(f"Applied schema to {db_path}")


def _ensure_timed_signal_columns(conn: sqlite3.Connection) -> None:
    columns = {
        "fire_at": "TEXT",
        "next_trigger_at": "TEXT",
        "rrule": "TEXT",
        "timezone": "TEXT",
        "fired_at": "TEXT",
        "attempt_count": "INTEGER NOT NULL DEFAULT 0",
        "attempts": "INTEGER NOT NULL DEFAULT 0",
        "last_error": "TEXT",
        "delivery_target": "TEXT",
    }
    for name, definition in columns.items():
        try:
            conn.execute(f"ALTER TABLE timed_signals ADD COLUMN {name} {definition}")
        except sqlite3.OperationalError:
            continue


def _ensure_timed_signal_statuses(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='timed_signals'"
    ).fetchone()
    if not row or not row[0]:
        return
    sql = str(row[0]).lower()
    if "check" in sql and "failed" in sql and "processing" in sql:
        return
    _rebuild_timed_signals(conn)


def _ensure_paired_device_columns(conn: sqlite3.Connection) -> None:
    columns = {
        "armed": "INTEGER NOT NULL DEFAULT 0",
        "armed_at": "TEXT",
        "armed_by": "TEXT",
        "armed_until": "TEXT",
        "token_hash": "TEXT",
        "token_expires_at": "TEXT",
    }
    for name, definition in columns.items():
        try:
            conn.execute(f"ALTER TABLE paired_devices ADD COLUMN {name} {definition}")
        except sqlite3.OperationalError:
            continue


def _ensure_pairing_columns(conn: sqlite3.Connection) -> None:
    # Ensure tables exist without forcing full migration logic.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pairing_requests (
          pairing_id   TEXT PRIMARY KEY,
          device_name  TEXT,
          challenge    TEXT,
          otp_hash     TEXT,
          status       TEXT NOT NULL,
          expires_at   TEXT NOT NULL,
          approved_via TEXT,
          approved_at  TEXT,
          created_at   TEXT NOT NULL
        ) STRICT
        """
    )


def _ensure_delivery_receipts_columns(conn: sqlite3.Connection) -> None:
    columns = [
        "receipt_id",
        "run_id",
        "pairing_id",
        "stage_id",
        "action_id",
        "skill",
        "channel",
        "status",
        "details_json",
        "created_at",
    ]
    try:
        rows = conn.execute("PRAGMA table_info(delivery_receipts)").fetchall()
    except sqlite3.OperationalError:
        return
    existing = {row[1] for row in rows}
    if not rows or "receipt_id" not in existing or "run_id" not in existing or "skill" not in existing:
        _rebuild_delivery_receipts(conn)


def _ensure_intent_specs_columns(conn: sqlite3.Connection) -> None:
    columns = {
        "intent_version": "TEXT NOT NULL DEFAULT '1.0.0'",
        "origin": "TEXT NOT NULL DEFAULT 'factory'",
        "parent_intent": "TEXT",
        "created_at": "TEXT NOT NULL DEFAULT '1970-01-01T00:00:00Z'",
        "updated_at": "TEXT NOT NULL DEFAULT '1970-01-01T00:00:00Z'",
    }
    for name, definition in columns.items():
        try:
            conn.execute(f"ALTER TABLE intent_specs ADD COLUMN {name} {definition}")
        except sqlite3.OperationalError:
            continue
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_intent_specs_category ON intent_specs (category, intent_name)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_intent_specs_parent ON intent_specs (parent_intent, intent_name)"
    )
    try:
        conn.execute(
            "UPDATE intent_specs SET created_at = datetime('now') WHERE created_at = '1970-01-01T00:00:00Z'"
        )
        conn.execute(
            "UPDATE intent_specs SET updated_at = datetime('now') WHERE updated_at = '1970-01-01T00:00:00Z'"
        )
    except sqlite3.OperationalError:
        pass


def _ensure_principals_constraints(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='principals'"
    ).fetchone()
    if not row or not row[0]:
        return
    sql = str(row[0]).lower()
    if "check" in sql and "system" in sql and "office" in sql:
        return
    _rebuild_principals(conn)


def _ensure_users_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
          user_id      TEXT PRIMARY KEY,
          principal_id TEXT,
          display_name TEXT NOT NULL,
          role         TEXT,
          relationship TEXT,
          is_admin     INTEGER NOT NULL DEFAULT 0,
          is_active    INTEGER NOT NULL DEFAULT 1,
          onboarded_at TEXT,
          created_at   TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
        ) STRICT
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_users_principal ON users (principal_id)"
    )


def _ensure_prompt_template_columns(conn: sqlite3.Connection) -> None:
    columns = {
        "purpose": "TEXT NOT NULL DEFAULT 'general'",
    }
    for name, definition in columns.items():
        try:
            conn.execute(f"ALTER TABLE prompt_templates ADD COLUMN {name} {definition}")
        except sqlite3.OperationalError:
            continue


def _ensure_sandbox_directories(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sandbox_directories (
          alias       TEXT PRIMARY KEY,
          base_path   TEXT NOT NULL,
          description TEXT,
          enabled     INTEGER NOT NULL DEFAULT 1,
          created_at  TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
          CHECK (enabled IN (0,1))
        ) STRICT
        """
    )
    sandbox_root = Path(
        os.getenv("ALPHONSE_SANDBOX_ROOT") or "/tmp/alphonse-sandbox"
    ).resolve()
    telegram_base = (sandbox_root / "telegram_files").resolve()
    conn.execute(
        """
        INSERT OR IGNORE INTO sandbox_directories (alias, base_path, description, enabled)
        VALUES (?, ?, ?, 1)
        """,
        (
            "telegram_files",
            str(telegram_base),
            "Downloaded Telegram files sandbox",
        ),
    )


def _rebuild_delivery_receipts(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE delivery_receipts RENAME TO delivery_receipts_old")
    conn.execute(
        """
        CREATE TABLE delivery_receipts (
          receipt_id   TEXT PRIMARY KEY,
          run_id       TEXT,
          pairing_id   TEXT,
          stage_id     TEXT,
          action_id    TEXT,
          skill        TEXT,
          channel      TEXT,
          status       TEXT NOT NULL,
          details_json TEXT,
          created_at   TEXT NOT NULL
        ) STRICT
        """
    )
    conn.execute(
        """
        INSERT INTO delivery_receipts (
          receipt_id, run_id, pairing_id, stage_id, action_id, skill, channel, status,
          details_json, created_at
        )
        SELECT id, NULL, pairing_id, NULL, NULL, channel, channel, status, details_json, created_at
        FROM delivery_receipts_old
        """
    )
    conn.execute("DROP TABLE delivery_receipts_old")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS delivery_receipts (
          receipt_id   TEXT PRIMARY KEY,
          run_id       TEXT,
          pairing_id   TEXT,
          stage_id     TEXT,
          action_id    TEXT,
          skill        TEXT,
          channel      TEXT,
          status       TEXT NOT NULL,
          details_json TEXT,
          created_at   TEXT NOT NULL
        ) STRICT
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
          id            TEXT PRIMARY KEY,
          event_type    TEXT NOT NULL,
          correlation_id TEXT,
          payload_json  TEXT,
          created_at    TEXT NOT NULL
        ) STRICT
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS habits (
          habit_id           TEXT PRIMARY KEY,
          name               TEXT NOT NULL,
          trigger            TEXT NOT NULL,
          conditions_json    TEXT NOT NULL,
          plan_json          TEXT NOT NULL,
          version            INTEGER NOT NULL,
          enabled            INTEGER NOT NULL DEFAULT 1,
          created_at         TEXT NOT NULL,
          updated_at         TEXT NOT NULL,
          success_count      INTEGER NOT NULL DEFAULT 0,
          fail_count         INTEGER NOT NULL DEFAULT 0,
          last_success_at    TEXT,
          last_fail_at       TEXT,
          menu_snapshot_hash TEXT
        ) STRICT
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS plan_runs (
          run_id         TEXT PRIMARY KEY,
          habit_id       TEXT,
          plan_id        TEXT NOT NULL,
          trigger        TEXT NOT NULL,
          correlation_id TEXT NOT NULL,
          status         TEXT NOT NULL,
          resolution     TEXT,
          resolved_via   TEXT,
          started_at     TEXT NOT NULL,
          ended_at       TEXT,
          state_json     TEXT,
          scheduled_json TEXT,
          plan_json      TEXT NOT NULL
        ) STRICT
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_habits_trigger_enabled ON habits (trigger, enabled)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_plan_runs_correlation ON plan_runs (correlation_id)"
    )


def _rebuild_timed_signals(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE timed_signals RENAME TO timed_signals_old")
    conn.execute(
        """
        CREATE TABLE timed_signals (
          id              TEXT PRIMARY KEY,
          trigger_at      TEXT NOT NULL,
          fire_at         TEXT,
          next_trigger_at TEXT,
          rrule           TEXT,
          timezone        TEXT,
          status          TEXT NOT NULL DEFAULT 'pending',
          fired_at        TEXT,
          attempt_count   INTEGER NOT NULL DEFAULT 0,
          attempts        INTEGER NOT NULL DEFAULT 0,
          last_error      TEXT,
          signal_type     TEXT NOT NULL,
          payload         TEXT,
          target          TEXT,
          delivery_target TEXT,
          origin          TEXT,
          correlation_id  TEXT,
          created_at      TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
          CHECK (status IN ('pending', 'processing', 'fired', 'failed', 'cancelled', 'error', 'skipped', 'dispatched'))
        ) STRICT
        """
    )
    columns = [row[1] for row in conn.execute("PRAGMA table_info(timed_signals_old)")]
    has_attempts = "attempts" in columns
    has_attempt_count = "attempt_count" in columns
    has_last_error = "last_error" in columns
    attempts_expr = "attempts" if has_attempts else "attempt_count" if has_attempt_count else "0"
    last_error_expr = "last_error" if has_last_error else "NULL"
    attempt_count_expr = "attempt_count" if has_attempt_count else "0"
    has_fire_at = "fire_at" in columns
    has_delivery_target = "delivery_target" in columns
    fire_at_expr = "fire_at" if has_fire_at else "trigger_at"
    delivery_target_expr = "delivery_target" if has_delivery_target else "target"
    conn.execute(
        f"""
        INSERT INTO timed_signals (
          id, trigger_at, fire_at, next_trigger_at, rrule, timezone, status, fired_at,
          attempt_count, attempts, last_error, signal_type, payload, target, delivery_target, origin,
          correlation_id, created_at, updated_at
        )
        SELECT
          id, trigger_at, {fire_at_expr}, next_trigger_at, rrule, timezone, status, fired_at,
          COALESCE({attempt_count_expr}, 0),
          COALESCE({attempts_expr}, 0),
          {last_error_expr}, signal_type, payload, target, {delivery_target_expr}, origin,
          correlation_id, created_at, updated_at
        FROM timed_signals_old
        """
    )
    conn.execute("DROP TABLE timed_signals_old")


def _rebuild_principals(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("ALTER TABLE principals RENAME TO principals_old")
    conn.execute(
        """
        CREATE TABLE principals (
          principal_id   TEXT PRIMARY KEY,
          principal_type TEXT NOT NULL,
          channel_type   TEXT,
          channel_id     TEXT,
          display_name   TEXT,
          created_at     TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at     TEXT NOT NULL DEFAULT (datetime('now')),
          CHECK (principal_type IN ('person', 'channel_chat', 'household', 'office', 'system'))
        ) STRICT
        """
    )
    conn.execute(
        """
        INSERT INTO principals (
          principal_id, principal_type, channel_type, channel_id, display_name, created_at, updated_at
        )
        SELECT
          principal_id, principal_type, channel_type, channel_id, display_name, created_at, updated_at
        FROM principals_old
        """
    )
    conn.execute("DROP TABLE principals_old")
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_principals_channel_unique
          ON principals (channel_type, channel_id)
          WHERE channel_type IS NOT NULL AND channel_id IS NOT NULL
        """
    )
    conn.execute("PRAGMA foreign_keys = ON")


if __name__ == "__main__":
    main()
