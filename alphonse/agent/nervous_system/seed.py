from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

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
    admin_user_id = _resolve_bootstrap_admin_user_id(conn)
    conn.execute(
        f"""
        INSERT INTO users (
          user_id, display_name, role, relationship, is_admin, is_active,
          onboarded_at, created_at, updated_at
        ) VALUES (
          ?, ?, ?, ?, 1, 1, {now}, {now}, {now}
        )
        ON CONFLICT(user_id) DO UPDATE SET
          display_name = excluded.display_name,
          role = excluded.role,
          relationship = excluded.relationship,
          is_admin = excluded.is_admin,
          is_active = excluded.is_active,
          updated_at = {now}
        """,
        (
            admin_user_id,
            BOOTSTRAP_ADMIN_DISPLAY_NAME,
            "owner",
            "self",
        ),
    )
    conn.execute(
        f"""
        INSERT INTO channels_users (
          mapping_id, user_id, channel_id, channel_user_id, is_active, created_at, updated_at
        ) VALUES (
          'resolver-bootstrap-cli-admin', ?, ?, ?, 1, {now}, {now}
        )
        ON CONFLICT(channel_id, channel_user_id) DO UPDATE SET
          user_id = excluded.user_id,
          is_active = excluded.is_active,
          updated_at = {now}
        """,
        (
            admin_user_id,
            BOOTSTRAP_CLI_SERVICE_ID,
            BOOTSTRAP_CLI_SERVICE_USER_ID,
        ),
    )


def _resolve_bootstrap_admin_user_id(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        """
        SELECT u.user_id
        FROM channels_users cu
        JOIN users u ON u.user_id = cu.user_id
        WHERE cu.channel_id = ?
          AND cu.channel_user_id = ?
          AND cu.is_active = 1
          AND u.is_admin = 1
          AND u.is_active = 1
        ORDER BY u.updated_at DESC
        LIMIT 1
        """,
        (BOOTSTRAP_CLI_SERVICE_ID, BOOTSTRAP_CLI_SERVICE_USER_ID),
    ).fetchone()
    if row and row[0]:
        return str(row[0])

    row = conn.execute(
        """
        SELECT user_id
        FROM users
        WHERE is_admin = 1 AND is_active = 1
        ORDER BY updated_at DESC
        LIMIT 1
        """
    ).fetchone()
    if row and row[0]:
        return str(row[0])

    return str(uuid.uuid4())
