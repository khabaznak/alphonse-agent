from __future__ import annotations

from pathlib import Path

from alphonse.agent import identity
from alphonse.agent.nervous_system.access_requests import get_access_request
from alphonse.agent.nervous_system.access_requests import upsert_access_request
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.telegram_chat_access import can_deliver_to_chat
from alphonse.agent.tools.access_request_tools import AccessRequestsTool
from alphonse.agent.tools.user_management_tools import UsersManageTool


def _prepare_db(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    identity.upsert_user(
        {
            "user_id": "admin-1",
            "display_name": "Admin",
            "is_admin": True,
            "is_active": True,
        }
    )
    identity.upsert_service_user_id(
        user_id="admin-1",
        service_id=2,
        service_user_id="111",
        is_active=True,
    )


def _admin_state() -> dict[str, str]:
    return {"service_key": "telegram", "incoming_user_id": "111", "channel_target": "111"}


def test_users_manage_invite_without_provider_id_creates_pending_claim(tmp_path: Path, monkeypatch) -> None:
    _prepare_db(tmp_path, monkeypatch)

    result = UsersManageTool().execute(
        action="invite",
        display_name="Maria Perez",
        relationship="sister",
        provider_key="telegram",
        state=_admin_state(),
    )

    assert result["exception"] is None
    output = result["output"]
    assert output["status"] == "pending_claim"
    request = get_access_request(output["request_id"])
    assert request
    assert request["kind"] == "user"
    assert request["status"] == "pending"
    assert request["claimed_user_id"] == output["user_id"]


def test_users_manage_register_from_contact_binds_provider_id(tmp_path: Path, monkeypatch) -> None:
    _prepare_db(tmp_path, monkeypatch)

    result = UsersManageTool().execute(
        action="register_from_contact",
        contact={"user_id": 222, "first_name": "Maria", "last_name": "Perez"},
        provider_key="telegram",
        state=_admin_state(),
    )

    assert result["exception"] is None
    output = result["output"]
    assert output["status"] == "registered"
    assert identity.resolve_user_id(service_id=2, service_user_id="222") == output["user_id"]


def test_users_manage_mutations_require_admin(tmp_path: Path, monkeypatch) -> None:
    _prepare_db(tmp_path, monkeypatch)

    result = UsersManageTool().execute(
        action="invite",
        display_name="Maria Perez",
        provider_key="telegram",
        state={"service_key": "telegram", "incoming_user_id": "999"},
    )

    assert result["exception"] is not None
    assert result["exception"]["code"] == "permission_denied"


def test_access_requests_approve_group_chat(tmp_path: Path, monkeypatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    request_id = upsert_access_request(
        {
            "kind": "chat",
            "provider_key": "telegram",
            "channel_target": "-100123",
            "display_name": "Family",
            "status": "pending",
            "metadata": {"chat_type": "group"},
        }
    )

    result = AccessRequestsTool().execute(action="approve", request_id=request_id, state=_admin_state())

    assert result["exception"] is None
    assert can_deliver_to_chat("-100123") is True
    request = get_access_request(request_id)
    assert request
    assert request["status"] == "approved"


def test_access_requests_list_and_show_pending_requests(tmp_path: Path, monkeypatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    request_id = upsert_access_request(
        {
            "kind": "user",
            "provider_key": "telegram",
            "provider_user_id": "222",
            "channel_target": "222",
            "display_name": "Maria",
            "status": "pending",
        }
    )
    tool = AccessRequestsTool()

    listed = tool.execute(action="list", status="pending", state=_admin_state())
    shown = tool.execute(action="show", request_id=request_id, state=_admin_state())

    assert listed["exception"] is None
    assert any(item["request_id"] == request_id for item in listed["output"]["requests"])
    assert shown["exception"] is None
    assert shown["output"]["request"]["request_id"] == request_id


def test_access_requests_deny_does_not_create_access(tmp_path: Path, monkeypatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    request_id = upsert_access_request(
        {
            "kind": "chat",
            "provider_key": "telegram",
            "channel_target": "-100456",
            "display_name": "Unknown Group",
            "status": "pending",
            "metadata": {"chat_type": "group"},
        }
    )

    result = AccessRequestsTool().execute(action="deny", request_id=request_id, reason="not needed", state=_admin_state())

    assert result["exception"] is None
    assert can_deliver_to_chat("-100456") is False
    request = get_access_request(request_id)
    assert request
    assert request["status"] == "denied"
