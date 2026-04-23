"""Apply schema to the nervous system SQLite database."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from alphonse.agent.nervous_system.sandbox_dirs import default_sandbox_root


def default_db_path() -> Path:
    return Path(__file__).resolve().parent / "db" / "nerve-db"


def apply_schema(db_path: Path) -> None:
    schema_path = Path(__file__).resolve().parent / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_sql)
        _drop_capability_gap_tables(conn)
        _drop_ability_specs_table(conn)
        _hard_reset_timed_signals(conn)
        _ensure_scheduled_jobs_table(conn)
        _ensure_paired_device_columns(conn)
        _ensure_pairing_columns(conn)
        _ensure_intent_specs_columns(conn)
        _ensure_principals_constraints(conn)
        _ensure_users_table(conn)
        _ensure_services_registry(conn)
        _ensure_voice_profiles_table(conn)
        _ensure_prompt_template_columns(conn)
        _ensure_prompt_artifacts_table(conn)
        _ensure_operational_facts_table(conn)
        _ensure_sandbox_directories(conn)
        _ensure_telegram_chat_access_table(conn)
        _ensure_telegram_pending_invites_table(conn)
        _ensure_telegram_invite_columns(conn)


def main() -> None:
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    apply_schema(db_path)
    print(f"Applied schema to {db_path}")


def _drop_capability_gap_tables(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS gap_tasks")
    conn.execute("DROP TABLE IF EXISTS gap_proposals")
    conn.execute("DROP TABLE IF EXISTS capability_gaps")


def _drop_ability_specs_table(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS ability_specs")


def _hard_reset_timed_signals(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS timed_signals")
    conn.execute(
        """
        CREATE TABLE timed_signals (
          id              TEXT PRIMARY KEY,
          trigger_at      TEXT NOT NULL,
          timezone        TEXT,
          status          TEXT NOT NULL DEFAULT 'pending',
          fired_at        TEXT,
          signal_type     TEXT NOT NULL,
          payload         TEXT,
          target          TEXT,
          origin          TEXT,
          correlation_id  TEXT,
          created_at      TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
          CHECK (status IN ('pending', 'fired', 'failed', 'cancelled', 'error'))
        ) STRICT
        """
    )


def _ensure_scheduled_jobs_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scheduled_jobs (
          id            TEXT PRIMARY KEY,
          name          TEXT NOT NULL,
          prompt        TEXT,
          owner_id      TEXT NOT NULL,
          rrule         TEXT,
          retries       INTEGER NOT NULL DEFAULT 0,
          status        TEXT NOT NULL DEFAULT 'active',
          next_run_at   TEXT,
          timezone      TEXT,
          created_at    TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
          CHECK (status IN ('active', 'paused', 'failed', 'completed'))
        ) STRICT
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_owner ON scheduled_jobs (owner_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_next_run ON scheduled_jobs (next_run_at, status)"
    )


def _ensure_paired_device_columns(conn: sqlite3.Connection) -> None:
    columns = {
        "armed": "INTEGER NOT NULL DEFAULT 0",
        "armed_at": "TEXT",
        "armed_by": "TEXT",
        "armed_until": "TEXT",
        "token_hash": "TEXT",
        "token_expires_at": "TEXT",
    }
    _ensure_columns(conn, table="paired_devices", columns=columns)


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


def _ensure_intent_specs_columns(conn: sqlite3.Connection) -> None:
    columns = {
        "intent_version": "TEXT NOT NULL DEFAULT '1.0.0'",
        "origin": "TEXT NOT NULL DEFAULT 'factory'",
        "parent_intent": "TEXT",
        "created_at": "TEXT NOT NULL DEFAULT '1970-01-01T00:00:00Z'",
        "updated_at": "TEXT NOT NULL DEFAULT '1970-01-01T00:00:00Z'",
    }
    _ensure_columns(conn, table="intent_specs", columns=columns)
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


def _ensure_services_registry(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS services (
          service_id         INTEGER PRIMARY KEY,
          service_key        TEXT NOT NULL UNIQUE,
          raw_user_key_field TEXT NOT NULL,
          name               TEXT NOT NULL,
          description        TEXT,
          created_at         TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at         TEXT NOT NULL DEFAULT (datetime('now'))
        ) STRICT
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_service_resolvers (
          resolver_id      TEXT PRIMARY KEY,
          user_id          TEXT NOT NULL,
          service_id       INTEGER NOT NULL,
          service_user_id  TEXT NOT NULL,
          is_active        INTEGER NOT NULL DEFAULT 1,
          created_at       TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at       TEXT NOT NULL DEFAULT (datetime('now')),
          CHECK (is_active IN (0,1))
        ) STRICT
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_user_service_resolver_unique
          ON user_service_resolvers (user_id, service_id)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_service_user_unique
          ON user_service_resolvers (service_id, service_user_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_service_resolvers_active
          ON user_service_resolvers (service_id, is_active)
        """
    )
    # Temporary fixed registry ids by request:
    # - Web UI: service_id=1
    # - Telegram: service_id=2
    # - CLI: service_id=3
    conn.execute(
        """
        INSERT INTO services (service_id, service_key, raw_user_key_field, name, description)
        VALUES (1, 'webui', 'user_id', 'Alphonse Web UI', 'Alphonse Web UI delivery')
        ON CONFLICT(service_id) DO UPDATE SET
          service_key = excluded.service_key,
          raw_user_key_field = excluded.raw_user_key_field,
          name = excluded.name,
          description = excluded.description,
          updated_at = datetime('now')
        """
    )
    conn.execute(
        """
        INSERT INTO services (service_id, service_key, raw_user_key_field, name, description)
        VALUES (2, 'telegram', 'chat_id', 'Telegram', 'Telegram chat delivery')
        ON CONFLICT(service_id) DO UPDATE SET
          service_key = excluded.service_key,
          raw_user_key_field = excluded.raw_user_key_field,
          name = excluded.name,
          description = excluded.description,
          updated_at = datetime('now')
        """
    )
    conn.execute(
        """
        INSERT INTO services (service_id, service_key, raw_user_key_field, name, description)
        VALUES (3, 'cli', 'cli_user_id', 'CLI', 'Alphonse CLI bootstrap communication')
        ON CONFLICT(service_id) DO UPDATE SET
          service_key = excluded.service_key,
          raw_user_key_field = excluded.raw_user_key_field,
          name = excluded.name,
          description = excluded.description,
          updated_at = datetime('now')
        """
    )


def _ensure_prompt_template_columns(conn: sqlite3.Connection) -> None:
    columns = {
        "purpose": "TEXT NOT NULL DEFAULT 'general'",
    }
    _ensure_columns(conn, table="prompt_templates", columns=columns)


def _ensure_voice_profiles_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS voice_profiles (
          profile_id          TEXT PRIMARY KEY,
          name                TEXT NOT NULL COLLATE NOCASE UNIQUE,
          source_sample_path  TEXT NOT NULL,
          backend             TEXT NOT NULL DEFAULT 'qwen',
          speaker_hint        TEXT,
          instruct            TEXT,
          is_default          INTEGER NOT NULL DEFAULT 0,
          status              TEXT NOT NULL DEFAULT 'pending',
          last_error          TEXT,
          created_at          TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
          CHECK (is_default IN (0,1)),
          CHECK (status IN ('pending', 'ready', 'error')),
          CHECK (backend IN ('qwen'))
        ) STRICT
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_voice_profiles_updated ON voice_profiles (updated_at DESC)"
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_voice_profiles_single_default
        ON voice_profiles (is_default) WHERE is_default = 1
        """
    )


def _ensure_prompt_artifacts_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prompt_artifacts (
          artifact_id       TEXT PRIMARY KEY,
          user_id           TEXT NOT NULL,
          source_instruction TEXT NOT NULL,
          agent_internal_prompt TEXT NOT NULL,
          language          TEXT,
          artifact_kind     TEXT NOT NULL,
          created_at        TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
        ) STRICT
        """
    )


def _ensure_operational_facts_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS operational_facts (
          id               TEXT PRIMARY KEY,
          key              TEXT NOT NULL,
          title            TEXT NOT NULL,
          fact_type        TEXT NOT NULL,
          summary          TEXT,
          content_json     TEXT,
          tags             TEXT,
          source           TEXT,
          stability        TEXT,
          importance       TEXT,
          status           TEXT,
          scope            TEXT NOT NULL DEFAULT 'private',
          created_by       TEXT NOT NULL,
          created_at       TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at       TEXT NOT NULL DEFAULT (datetime('now')),
          last_verified_at TEXT,
          confidence       REAL,
          CHECK (fact_type IN (
            'system_asset',
            'procedure',
            'workflow_rule',
            'location',
            'integration_note',
            'user_operational_preference'
          )),
          CHECK (scope IN ('private', 'global')),
          CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0))
        ) STRICT
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_operational_facts_key_unique ON operational_facts (key)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_operational_facts_created_by_scope ON operational_facts (created_by, scope)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_operational_facts_type_status_updated ON operational_facts (fact_type, status, updated_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_operational_facts_updated_at ON operational_facts (updated_at DESC)"
    )


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
    sandbox_root = default_sandbox_root()
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


def _ensure_telegram_chat_access_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS telegram_chat_access (
          chat_id       TEXT PRIMARY KEY,
          chat_type     TEXT NOT NULL,
          status        TEXT NOT NULL DEFAULT 'active',
          owner_user_id TEXT,
          policy        TEXT NOT NULL DEFAULT 'registered_private',
          created_at    TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
          revoked_at    TEXT,
          revoke_reason TEXT,
          CHECK (chat_type IN ('private', 'group', 'supergroup')),
          CHECK (status IN ('active', 'revoked', 'pending')),
          CHECK (policy IN ('registered_private', 'owner_managed_group'))
        ) STRICT
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_telegram_chat_access_status
          ON telegram_chat_access (status, updated_at)
        """
    )


def _ensure_telegram_pending_invites_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS telegram_pending_invites (
          chat_id             TEXT PRIMARY KEY,
          chat_type           TEXT,
          from_user_id        TEXT,
          from_user_username  TEXT,
          from_user_name      TEXT,
          last_message        TEXT,
          status              TEXT NOT NULL DEFAULT 'pending',
          created_at          TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
        ) STRICT
        """
    )


def _ensure_telegram_invite_columns(conn: sqlite3.Connection) -> None:
    columns = {
        "chat_type": "TEXT",
        "from_user_username": "TEXT",
    }
    _ensure_columns(conn, table="telegram_pending_invites", columns=columns)


def _ensure_columns(
    conn: sqlite3.Connection,
    *,
    table: str,
    columns: dict[str, str],
) -> None:
    existing = {
        str(row[1] or "")
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name, definition in columns.items():
        if name in existing:
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


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
