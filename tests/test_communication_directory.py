from __future__ import annotations

from pathlib import Path

from alphonse.agent.nervous_system import user_service_resolvers as resolvers
from alphonse.agent.nervous_system import users as users_store
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.services import communication_directory


def test_directory_resolves_service_and_user_mappings(tmp_path: Path, monkeypatch) -> None:
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
    resolvers.upsert_service_resolver(
        user_id="u-1",
        service_id=2,
        service_user_id="8553589429",
        is_active=True,
    )

    assert communication_directory.resolve_service_id("telegram") == 2
    assert communication_directory.resolve_service(2)["service_key"] == "telegram"
    assert communication_directory.resolve_service_key(2) == "telegram"
    assert communication_directory.resolve_user_id(service_id=2, service_user_id="8553589429") == "u-1"
    assert communication_directory.resolve_service_user_id(user_id="u-1", service_id=2) == "8553589429"
    assert communication_directory.resolve_delivery_target(user_id="u-1", service_id=2) == "8553589429"


def test_directory_returns_none_for_unknown_provider_or_mapping(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    assert communication_directory.resolve_service_id("does-not-exist") is None
    assert communication_directory.resolve_user_id(service_id=999, service_user_id="abc") is None
    assert communication_directory.resolve_service_user_id(user_id="u-missing", service_id=2) is None
