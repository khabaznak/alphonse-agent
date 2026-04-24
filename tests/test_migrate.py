from __future__ import annotations

import sqlite3
from pathlib import Path

from alphonse.agent.nervous_system.migrate import apply_schema


def test_apply_schema_drops_legacy_capability_gap_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "nerve-db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE capability_gaps (gap_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE gap_proposals (id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE gap_tasks (id TEXT PRIMARY KEY)")

    apply_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert "capability_gaps" not in names
    assert "gap_proposals" not in names
    assert "gap_tasks" not in names


def test_apply_schema_drops_legacy_ability_specs_table(tmp_path: Path) -> None:
    db_path = tmp_path / "nerve-db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE ability_specs (intent_name TEXT PRIMARY KEY)")

    apply_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert "ability_specs" not in names


def test_apply_schema_preserves_existing_channels_users_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "nerve-db"
    apply_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO users (
              user_id, display_name, role, relationship, is_admin, is_active,
              onboarded_at, created_at, updated_at
            ) VALUES (
              'e64b2111-35ba-41b1-a295-4ff04fd4cf58', 'Alex Gomez', 'owner', 'self', 1, 1,
              datetime('now'), datetime('now'), datetime('now')
            )
            """
        )
        conn.execute(
            """
            INSERT INTO channels_users (
              mapping_id, user_id, channel_id, channel_user_id, is_active, created_at, updated_at
            ) VALUES (
              'resolver-telegram-alex', 'e64b2111-35ba-41b1-a295-4ff04fd4cf58', 2, '8553589429', 1,
              datetime('now'), datetime('now')
            )
            """
        )
        conn.commit()

    apply_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT user_id
            FROM channels_users
            WHERE channel_id = 2 AND channel_user_id = '8553589429' AND is_active = 1
            LIMIT 1
            """
        ).fetchone()

    assert row is not None
    assert row[0] == "e64b2111-35ba-41b1-a295-4ff04fd4cf58"
