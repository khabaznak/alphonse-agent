from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.seed import (
    BOOTSTRAP_CLI_SERVICE_USER_ID,
    apply_seed,
)
from alphonse.agent.services import communication_directory


def test_apply_seed_creates_bootstrap_cli_admin_identity(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    apply_seed(db_path)

    with sqlite3.connect(db_path) as conn:
        service = conn.execute(
            "SELECT channel_id, channel_key FROM channels WHERE channel_key = 'cli'"
        ).fetchone()
        user = conn.execute(
            "SELECT user_id, is_admin, is_active FROM users WHERE is_admin = 1 AND is_active = 1"
        ).fetchone()
        resolver = conn.execute(
            "SELECT user_id, channel_user_id FROM channels_users WHERE channel_id = 3 AND channel_user_id = ?",
            (BOOTSTRAP_CLI_SERVICE_USER_ID,),
        ).fetchone()

    assert service is not None
    assert int(service[0]) == 3
    assert str(service[1]) == "cli"
    assert user is not None
    uuid.UUID(str(user[0]))
    assert str(user[0]) != "owner-1"
    assert int(user[1]) == 1
    assert int(user[2]) == 1
    assert resolver is not None
    assert str(resolver[0]) == str(user[0])
    assert str(resolver[1]) == BOOTSTRAP_CLI_SERVICE_USER_ID


def test_apply_seed_is_idempotent_for_bootstrap_cli_admin(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    apply_seed(db_path)
    apply_seed(db_path)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT u.user_id, cu.channel_user_id
            FROM users u
            JOIN channels_users cu ON cu.user_id = u.user_id
            WHERE u.is_admin = 1 AND u.is_active = 1 AND cu.channel_id = 3
            """
        ).fetchall()
    assert len(rows) == 1
    assert communication_directory.resolve_service_id("cli") == 3
    assert communication_directory.resolve_service_user_id(user_id=str(rows[0][0]), service_id=3) == BOOTSTRAP_CLI_SERVICE_USER_ID
