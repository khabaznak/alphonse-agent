from __future__ import annotations

from alphonse.agent.actions import handle_incoming_message as him
from alphonse.agent.actions.handle_incoming_message import IncomingContext


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
