from __future__ import annotations

from pathlib import Path

from alphonse.agent.nervous_system import user_service_resolvers as resolvers
from alphonse.agent.nervous_system import users as users_store
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.timed_store import list_timed_signals
from alphonse.agent.tools.user_contact_tools import (
    UserRegisterFromContactTool,
    UserRemoveFromContactTool,
)


def _prepare_db(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)


def _state_with_contact(*, sender_telegram_id: str, contact_telegram_id: str) -> dict[str, object]:
    return {
        "incoming_user_id": sender_telegram_id,
        "incoming_raw_message": {
            "provider_event": {
                "message": {
                    "contact": {
                        "user_id": int(contact_telegram_id),
                        "first_name": "Maria",
                        "last_name": "Perez",
                        "phone_number": "+5215512345678",
                    }
                }
            }
        },
        "locale": "es-MX",
        "timezone": "UTC",
        "correlation_id": "cid-contact-tool",
    }


def test_register_from_contact_requires_admin(tmp_path: Path, monkeypatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    users_store.upsert_user(
        {
            "user_id": "u-member",
            "display_name": "Member",
            "is_admin": False,
            "is_active": True,
        }
    )
    resolvers.upsert_service_resolver(
        user_id="u-member",
        service_id=2,
        service_user_id="111",
        is_active=True,
    )
    tool = UserRegisterFromContactTool()
    result = tool.execute(state=_state_with_contact(sender_telegram_id="111", contact_telegram_id="222"))

    assert result["status"] == "failed"
    assert result["error"]["code"] == "permission_denied"


def test_register_from_contact_admin_schedules_intro(tmp_path: Path, monkeypatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    users_store.upsert_user(
        {
            "user_id": "u-admin",
            "display_name": "Admin",
            "is_admin": True,
            "is_active": True,
        }
    )
    resolvers.upsert_service_resolver(
        user_id="u-admin",
        service_id=2,
        service_user_id="111",
        is_active=True,
    )
    tool = UserRegisterFromContactTool()
    result = tool.execute(
        role="family",
        relationship="sister",
        state=_state_with_contact(sender_telegram_id="111", contact_telegram_id="222"),
    )

    assert result["status"] == "ok"
    payload = result["result"]
    assert payload["telegram_user_id"] == "222"
    assert payload["scheduled_intro_signal_id"]

    internal_user_id = resolvers.resolve_internal_user_by_telegram_id("222")
    assert internal_user_id
    user = users_store.get_user(internal_user_id)
    assert user
    assert user["display_name"] == "Maria Perez"
    assert user["role"] == "family"
    assert user["relationship"] == "sister"

    timed = list_timed_signals(limit=10)
    assert any(
        str(item.get("target") or "") == "222" and str(item.get("signal_type") or "") == "reminder"
        for item in timed
    )


def test_remove_from_contact_deactivates_user(tmp_path: Path, monkeypatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    users_store.upsert_user(
        {
            "user_id": "u-admin",
            "display_name": "Admin",
            "is_admin": True,
            "is_active": True,
        }
    )
    resolvers.upsert_service_resolver(
        user_id="u-admin",
        service_id=2,
        service_user_id="111",
        is_active=True,
    )
    users_store.upsert_user(
        {
            "user_id": "u-target",
            "display_name": "Maria",
            "is_admin": False,
            "is_active": True,
        }
    )
    resolvers.upsert_service_resolver(
        user_id="u-target",
        service_id=2,
        service_user_id="222",
        is_active=True,
    )

    tool = UserRemoveFromContactTool()
    result = tool.execute(
        state={
            "incoming_user_id": "111",
            "incoming_raw_message": {
                "provider_event": {"message": {"contact": {"user_id": 222}}}
            },
        }
    )

    assert result["status"] == "ok"
    assert result["result"]["deactivated"] is True
    user = users_store.get_user("u-target")
    assert user
    assert user["is_active"] is False
    assert resolvers.resolve_internal_user_by_telegram_id("222") is None
