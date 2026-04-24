from __future__ import annotations

from alphonse.agent import identity
from alphonse.agent.actions.session_context import IncomingContext
from alphonse.agent.identity.session import resolve_session_user_id
from alphonse.agent.nervous_system import user_service_resolvers as resolvers
from alphonse.agent.nervous_system import users as users_store
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.tools.context_tools import GetUserDetailsTool


def test_identity_facade_resolves_canonical_user_and_service_mapping(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    users_store.upsert_user(
        {
            "user_id": "u-1",
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

    assert identity.get_user("u-1")["user_id"] == "u-1"
    assert identity.resolve_user_id(service_id=2, service_user_id="8553589429") == "u-1"
    assert identity.resolve_service_user_id(user_id="u-1", service_id=2) == "8553589429"
    assert identity.resolve_delivery_target(user_id="u-1", service_id=2) == "8553589429"


def test_identity_facade_resolves_session_user_from_service_mapping(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    users_store.upsert_user(
        {
            "user_id": "u-telegram",
            "display_name": "Telegram User",
            "is_active": True,
        }
    )
    resolvers.upsert_service_resolver(
        user_id="u-telegram",
        service_id=2,
        service_user_id="777001",
        is_active=True,
    )
    incoming = IncomingContext(
        channel_type="telegram",
        address="777001",
        person_id=None,
        correlation_id="corr-1",
    )
    payload = {
        "provider": "telegram",
        "user_id": "777001",
        "metadata": {"service_key": "telegram"},
    }

    assert resolve_session_user_id(incoming=incoming, payload=payload) == "u-telegram"


def test_get_user_details_returns_canonical_user_id(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    users_store.upsert_user(
        {
            "user_id": "u-cli",
            "display_name": "CLI User",
            "is_active": True,
        }
    )
    tool = GetUserDetailsTool()
    result = tool.execute(
        task_record=TaskRecord(task_id="task-1", user_id="u-cli", correlation_id="corr-1")
    )

    assert result["exception"] is None
    output = result["output"]
    assert output["user_id"] == "u-cli"
