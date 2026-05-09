from __future__ import annotations

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent import identity as users_store
from alphonse.agent.nervous_system import user_service_resolvers as resolvers
from alphonse.agent.nervous_system.services import TELEGRAM_SERVICE_ID
from alphonse.agent.tools.registry import build_default_tool_registry
from alphonse.agent.tools.registry import planner_tool_schemas
from alphonse.agent.tools.user_management_tools import UsersManageTool


def _seed_user(display_name: str, telegram_id: str) -> str:
    user_id = users_store.upsert_user(
        {
            "display_name": display_name,
            "is_active": True,
            "is_admin": False,
        }
    )
    resolvers.upsert_service_resolver(
        user_id=user_id,
        service_id=TELEGRAM_SERVICE_ID,
        service_user_id=telegram_id,
        is_active=True,
    )
    return user_id


def test_users_manage_tool_registered_and_legacy_tools_removed() -> None:
    runtime = build_default_tool_registry()
    assert runtime.get("users.manage") is not None
    assert runtime.get("users.search") is None
    assert runtime.get("users.register_from_contact") is None
    assert runtime.get("users.remove_from_contact") is None
    schema_names = {
        str(item.get("function", {}).get("name") or "")
        for item in planner_tool_schemas(runtime)
        if isinstance(item, dict)
    }
    assert "users.manage" in schema_names
    assert "users.search" not in schema_names


def test_user_search_returns_matches_with_telegram_ids(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    _seed_user("Gabriela", "777001")
    _seed_user("Gabrielle", "777002")
    tool = UsersManageTool()
    result = tool.execute(action="search", query="gabr", limit=10, active_only=True)
    assert result["exception"] is None
    users = result["output"]["users"]
    assert len(users) >= 2
    assert all("telegram_user_id" not in item for item in users)


def test_user_search_requires_query(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    tool = UsersManageTool()
    result = tool.execute(action="search", query="")
    assert result["exception"] is None
    assert isinstance(result["output"]["users"], list)
