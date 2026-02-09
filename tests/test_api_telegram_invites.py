from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.telegram_invites import upsert_invite
from alphonse.infrastructure.api import app


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)


def test_api_telegram_invites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    upsert_invite(
        {
            "chat_id": "-123",
            "from_user_id": "gaby",
            "from_user_name": "Gaby",
            "last_message": "hello",
            "status": "pending",
        }
    )
    client = TestClient(app)
    listed = client.get("/agent/telegram/invites?status=pending")
    assert listed.status_code == 200
    assert listed.json()["items"]

    item = client.get("/agent/telegram/invites/-123")
    assert item.status_code == 200
    assert item.json()["item"]["chat_id"] == "-123"

    updated = client.post("/agent/telegram/invites/-123/status", json={"status": "approved"})
    assert updated.status_code == 200
    assert updated.json()["item"]["status"] == "approved"
