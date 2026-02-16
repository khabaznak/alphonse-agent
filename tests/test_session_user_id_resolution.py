from __future__ import annotations

from alphonse.agent.actions import handle_incoming_message as him
from alphonse.agent.actions.handle_incoming_message import IncomingContext


def test_resolve_session_user_id_prefers_db_user_from_principal_id(monkeypatch) -> None:
    incoming = IncomingContext(
        channel_type="telegram",
        address="8553589429",
        person_id=None,
        correlation_id="corr-1",
    )
    payload = {
        "channel": "telegram",
        "chat_id": "8553589429",
    }
    monkeypatch.setattr(him, "principal_id_for_incoming", lambda _: "principal-123")
    monkeypatch.setattr(
        him.users_store,
        "get_user_by_principal_id",
        lambda principal_id: {"user_id": "user-e64"} if principal_id == "principal-123" else None,
    )

    result = him._resolve_session_user_id(incoming=incoming, payload=payload)
    assert result == "user-e64"


def test_resolve_session_user_id_prefers_db_surrogate_from_display_name(monkeypatch) -> None:
    incoming = IncomingContext(
        channel_type="webui",
        address="webui",
        person_id=None,
        correlation_id="corr-1",
    )
    payload = {
        "channel": "webui",
        "target": "webui",
        "metadata": {"raw": {"metadata": {"user_name": "Alex"}}},
    }
    monkeypatch.setattr(
        him.users_store,
        "get_user_by_display_name",
        lambda name: {"user_id": "user-42"} if name == "Alex" else None,
    )

    result = him._resolve_session_user_id(incoming=incoming, payload=payload)
    assert result == "user-42"


def test_resolve_session_user_id_falls_back_to_name_when_db_has_no_match() -> None:
    incoming = IncomingContext(
        channel_type="webui",
        address="webui",
        person_id=None,
        correlation_id="corr-1",
    )
    payload = {
        "channel": "webui",
        "target": "webui",
        "metadata": {"raw": {"metadata": {"user_name": "Alex"}}},
    }

    result = him._resolve_session_user_id(incoming=incoming, payload=payload)
    assert result == "name:alex"


def test_resolve_session_user_id_extracts_display_name_from_provider_event() -> None:
    incoming = IncomingContext(
        channel_type="telegram",
        address="8553589429",
        person_id=None,
        correlation_id="corr-1",
    )
    payload = {
        "channel": "telegram",
        "chat_id": "8553589429",
        "provider_event": {
            "message": {
                "from": {"first_name": "Alex"},
            }
        },
    }

    result = him._resolve_session_user_id(incoming=incoming, payload=payload)
    assert result == "name:alex"


def test_resolve_session_user_id_telegram_prefers_name_over_numeric_user_id() -> None:
    incoming = IncomingContext(
        channel_type="telegram",
        address="8553589429",
        person_id=None,
        correlation_id="corr-1",
    )
    payload = {
        "channel": "telegram",
        "chat_id": "8553589429",
        "user_id": "8553589429",
        "user_name": "Alex",
    }

    result = him._resolve_session_user_id(incoming=incoming, payload=payload)
    assert result == "name:alex"
