from __future__ import annotations

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system import users as users_store
from alphonse.agent.nervous_system import user_service_resolvers as resolvers
from alphonse.agent.nervous_system.services import TELEGRAM_SERVICE_ID
from alphonse.agent.cognition.tool_schemas import llm_tool_schemas
from alphonse.agent.tools.registry import build_default_tool_registry
from alphonse.agent.tools.user_contact_tools import UserSearchTool


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


def test_user_search_tool_registered() -> None:
    runtime = build_default_tool_registry()
    assert runtime.get("user_search") is not None
    schema_names = {
        str(item.get("function", {}).get("name") or "")
        for item in llm_tool_schemas(runtime)
        if isinstance(item, dict)
    }
    assert "user_search" in schema_names


def test_user_search_returns_matches_with_telegram_ids(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    _seed_user("Gabriela", "777001")
    _seed_user("Gabrielle", "777002")
    tool = UserSearchTool()
    result = tool.execute(query="gabr", limit=10, active_only=True)
    assert result["status"] == "ok"
    users = result["result"]["users"]
    assert len(users) >= 2
    ids = {str(item.get("telegram_user_id") or "") for item in users}
    assert "777001" in ids
    assert "777002" in ids


def test_user_search_requires_query(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    tool = UserSearchTool()
    failed = tool.execute(query="")
    assert failed["status"] == "failed"
    assert str((failed.get("error") or {}).get("code") or "") == "missing_query"
