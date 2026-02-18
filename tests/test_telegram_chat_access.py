from __future__ import annotations

from pathlib import Path

from alphonse.agent.nervous_system import users as users_store
from alphonse.agent.nervous_system import user_service_resolvers as resolvers
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.telegram_chat_access import (
    can_deliver_to_chat,
    evaluate_inbound_access,
    get_chat_access,
    revoke_chat_access,
    upsert_chat_access,
)
from alphonse.agent.nervous_system.telegram_invites import (
    upsert_invite,
    update_invite_status,
)


def _prepare_db(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)


def test_private_inbound_allowed_for_registered_user(tmp_path: Path, monkeypatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    users_store.upsert_user(
        {"user_id": "u-1", "display_name": "Alex", "is_active": True}
    )
    resolvers.upsert_service_resolver(
        user_id="u-1",
        service_id=2,
        service_user_id="8553589429",
        is_active=True,
    )

    decision = evaluate_inbound_access(
        chat_id="8553589429",
        chat_type="private",
        from_user_id="8553589429",
    )

    assert decision.allowed is True
    assert decision.reason == "registered_private"


def test_private_inbound_denied_for_unknown_user(tmp_path: Path, monkeypatch) -> None:
    _prepare_db(tmp_path, monkeypatch)

    decision = evaluate_inbound_access(
        chat_id="9999",
        chat_type="private",
        from_user_id="9999",
    )

    assert decision.allowed is False
    assert decision.emit_invite is True
    assert decision.reason == "private_not_registered"


def test_group_invite_approval_provisions_access(tmp_path: Path, monkeypatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    users_store.upsert_user(
        {"user_id": "owner-1", "display_name": "Alex", "is_active": True}
    )
    resolvers.upsert_service_resolver(
        user_id="owner-1",
        service_id=2,
        service_user_id="8553589429",
        is_active=True,
    )
    upsert_invite(
        {
            "chat_id": "-100123",
            "chat_type": "group",
            "from_user_id": "8553589429",
            "from_user_name": "Alex",
            "status": "pending",
        }
    )

    update_invite_status("-100123", "approved")
    access = get_chat_access("-100123")

    assert access
    assert access["status"] == "active"
    assert access["policy"] == "owner_managed_group"
    assert access["owner_user_id"] == "owner-1"


def test_can_deliver_to_chat_for_group_access(tmp_path: Path, monkeypatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    upsert_chat_access(
        {
            "chat_id": "-100200",
            "chat_type": "group",
            "status": "active",
            "policy": "owner_managed_group",
        }
    )
    assert can_deliver_to_chat("-100200") is True

    revoke_chat_access("-100200", "owner_missing")
    assert can_deliver_to_chat("-100200") is False
