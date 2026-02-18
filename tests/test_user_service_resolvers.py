from __future__ import annotations

from pathlib import Path

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system import user_service_resolvers as resolvers
from alphonse.agent.nervous_system import users as users_store


def test_upsert_and_resolve_service_user_id(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    users_store.upsert_user(
        {
            "user_id": "u-1",
            "principal_id": "p-1",
            "display_name": "Alex",
            "is_active": True,
        }
    )
    _ = resolvers.upsert_service_resolver(
        user_id="u-1",
        service_id=2,
        service_user_id="8553589429",
        is_active=True,
    )

    resolved = resolvers.resolve_service_user_id(user_id="u-1", service_id=2)
    assert resolved == "8553589429"


def test_resolve_telegram_chat_id_for_internal_and_display_name(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    users_store.upsert_user(
        {
            "user_id": "u-2",
            "principal_id": "p-2",
            "display_name": "Alex",
            "is_active": True,
        }
    )
    _ = resolvers.upsert_service_resolver(
        user_id="u-2",
        service_id=2,
        service_user_id="8553589429",
        is_active=True,
    )

    assert resolvers.resolve_telegram_chat_id_for_user("u-2") == "8553589429"
    assert resolvers.resolve_telegram_chat_id_for_user("Alex") == "8553589429"
    assert resolvers.resolve_telegram_chat_id_for_user("8553589429") == "8553589429"


def test_resolve_internal_user_by_telegram_id(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    users_store.upsert_user(
        {
            "user_id": "u-3",
            "principal_id": "p-3",
            "display_name": "Rex",
            "is_active": True,
        }
    )
    _ = resolvers.upsert_service_resolver(
        user_id="u-3",
        service_id=2,
        service_user_id="123456789",
        is_active=True,
    )

    assert resolvers.resolve_internal_user_by_telegram_id("123456789") == "u-3"
