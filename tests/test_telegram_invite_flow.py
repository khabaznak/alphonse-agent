from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.telegram_invites import get_invite, list_invites
from alphonse.agent.nervous_system.senses.telegram import TelegramSense
from alphonse.agent.nervous_system.senses.bus import Bus, Signal


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)


def test_invite_request_creates_pending_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    bus = Bus()
    sense = TelegramSense()
    sense._bus = bus  # type: ignore[attr-defined]

    sense._on_signal(  # type: ignore[attr-defined]
        Signal(
            type="external.telegram.invite_request",
            payload={
                "chat_id": "-123",
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
    items = list_invites(status="pending")
    assert items
