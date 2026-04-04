from __future__ import annotations

import sqlite3
from pathlib import Path

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.seed import (
    BOOTSTRAP_ADMIN_USER_ID,
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
            "SELECT service_id, service_key FROM services WHERE service_key = 'cli'"
        ).fetchone()
        user = conn.execute(
            "SELECT user_id, is_admin, is_active FROM users WHERE user_id = ?",
            (BOOTSTRAP_ADMIN_USER_ID,),
        ).fetchone()
        resolver = conn.execute(
            "SELECT service_user_id FROM user_service_resolvers WHERE user_id = ? AND service_id = 3",
            (BOOTSTRAP_ADMIN_USER_ID,),
        ).fetchone()

    assert service is not None
    assert int(service[0]) == 3
    assert str(service[1]) == "cli"
    assert user is not None
    assert str(user[0]) == BOOTSTRAP_ADMIN_USER_ID
    assert int(user[1]) == 1
    assert int(user[2]) == 1
    assert resolver is not None
    assert str(resolver[0]) == BOOTSTRAP_CLI_SERVICE_USER_ID


def test_apply_seed_is_idempotent_for_bootstrap_cli_admin(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    apply_seed(db_path)
    apply_seed(db_path)

    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM user_service_resolvers WHERE user_id = ? AND service_id = 3",
            (BOOTSTRAP_ADMIN_USER_ID,),
        ).fetchone()
    assert count is not None
    assert int(count[0]) == 1
    assert communication_directory.resolve_service_id("cli") == 3
    assert communication_directory.resolve_service_user_id(user_id=BOOTSTRAP_ADMIN_USER_ID, service_id=3) == BOOTSTRAP_CLI_SERVICE_USER_ID
