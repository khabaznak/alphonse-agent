from __future__ import annotations

import sqlite3
from pathlib import Path

BOOTSTRAP_ADMIN_USER_ID = "owner-1"
BOOTSTRAP_ADMIN_PRINCIPAL_ID = "owner-1"
BOOTSTRAP_ADMIN_DISPLAY_NAME = "Alex"
BOOTSTRAP_CLI_SERVICE_ID = 3
BOOTSTRAP_CLI_SERVICE_USER_ID = "cli-admin"


def apply_seed(db_path: Path) -> None:
    seed_path = Path(__file__).resolve().parent / "seed.sql"
    with sqlite3.connect(db_path) as conn:
        if seed_path.exists():
            seed_sql = seed_path.read_text(encoding="utf-8")
            conn.executescript(seed_sql)
        _seed_bootstrap_admin(conn)


def _seed_bootstrap_admin(conn: sqlite3.Connection) -> None:
    now = "datetime('now')"
    conn.execute(
        f"""
        INSERT INTO users (
          user_id, principal_id, display_name, role, relationship, is_admin, is_active,
          onboarded_at, created_at, updated_at
        ) VALUES (
          ?, ?, ?, ?, ?, 1, 1, {now}, {now}, {now}
        )
        ON CONFLICT(user_id) DO UPDATE SET
          principal_id = excluded.principal_id,
          display_name = excluded.display_name,
          role = excluded.role,
          relationship = excluded.relationship,
          is_admin = excluded.is_admin,
          is_active = excluded.is_active,
          updated_at = {now}
        """,
        (
            BOOTSTRAP_ADMIN_USER_ID,
            BOOTSTRAP_ADMIN_PRINCIPAL_ID,
            BOOTSTRAP_ADMIN_DISPLAY_NAME,
            "owner",
            "self",
        ),
    )
    conn.execute(
        f"""
        INSERT OR IGNORE INTO principals (
          principal_id, principal_type, display_name, created_at, updated_at
        ) VALUES (
          ?, 'person', ?, {now}, {now}
        )
        """,
        (
            BOOTSTRAP_ADMIN_PRINCIPAL_ID,
            BOOTSTRAP_ADMIN_DISPLAY_NAME,
        ),
    )
    conn.execute(
        f"""
        INSERT INTO user_service_resolvers (
          resolver_id, user_id, service_id, service_user_id, is_active, created_at, updated_at
        ) VALUES (
          'resolver-bootstrap-cli-admin', ?, ?, ?, 1, {now}, {now}
        )
        ON CONFLICT(user_id, service_id) DO UPDATE SET
          service_user_id = excluded.service_user_id,
          is_active = excluded.is_active,
          updated_at = {now}
        """,
        (
            BOOTSTRAP_ADMIN_USER_ID,
            BOOTSTRAP_CLI_SERVICE_ID,
            BOOTSTRAP_CLI_SERVICE_USER_ID,
        ),
    )
