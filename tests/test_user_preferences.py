from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.cognition.preferences.store import (
    delete_user_preference,
    get_user_preference,
    get_with_fallback,
    set_user_preference,
)
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system import users as users_store


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    users_store.upsert_user({"user_id": "u-1", "display_name": "Alex", "is_active": True})


def test_user_preferences_store_and_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_db(tmp_path, monkeypatch)

    set_user_preference("u-1", "locale", "es-MX", source="user")

    assert get_user_preference("u-1", "locale") == "es-MX"
    assert get_with_fallback("u-1", "locale", "en-US") == "es-MX"


def test_user_preferences_delete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_db(tmp_path, monkeypatch)

    set_user_preference("u-1", "tone", "formal", source="user")
    assert delete_user_preference("u-1", "tone") is True
    assert get_user_preference("u-1", "tone") is None
