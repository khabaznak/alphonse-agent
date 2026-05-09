from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.access_requests import get_access_request
from alphonse.agent.nervous_system.telegram_invites import get_invite, list_invites
from alphonse.agent.nervous_system.senses.telegram import TelegramSense
from alphonse.agent.nervous_system.senses.bus import Signal


class _FakeBus:
    def __init__(self) -> None:
        self.emitted: list[Signal] = []

    def emit(self, signal: Signal) -> None:
        self.emitted.append(signal)


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)


def test_invite_request_creates_pending_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    bus = _FakeBus()
    sense = TelegramSense()
    sense._bus = bus  # type: ignore[attr-defined]

    sense._on_signal(  # type: ignore[attr-defined]
        Signal(
            type="external.telegram.invite_request",
            payload={
                "chat_id": "-123",
                "chat_type": "group",
                "request_kind": "chat",
                "from_user": "gaby",
                "from_user_name": "Gaby",
                "text": "hello",
            },
            source="telegram",
        )
    )

    invite = get_invite("-123")
    assert invite
    assert invite["status"] == "pending"
    assert invite["from_user_name"] == "Gaby"
    request = get_access_request("chat:telegram:-123")
    assert request
    assert request["kind"] == "chat"
    assert request["status"] == "pending"
    items = list_invites(status="pending")
    assert items
    assert len(bus.emitted) == 1
    emitted = bus.emitted[0]
    assert emitted.type == "sense.telegram.message.user.received"
    assert emitted.source == "telegram"
    payload = emitted.payload
    assert payload["contract_type"] == "canonical_inbound_event"
    assert payload["service_key"] == "telegram"
    assert payload["provider_user_id_from"] == "gaby"
    assert payload["provider_message_id"] == "-123"
    assert payload["channel_target"] == "-123"
    assert payload["event_kind"] == "invite"
    assert payload["provider_raw_message"]["chat_id"] == "-123"
