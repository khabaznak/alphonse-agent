from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.infrastructure.api import app


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)


def test_tool_config_api_crud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    client = TestClient(app)

    created = client.post(
        "/agent/tool-configs",
        json={
            "tool_key": "geocoder",
            "name": "google",
            "config": {"api_key": "test"},
            "is_active": True,
        },
    )
    assert created.status_code == 201
    config_id = created.json()["item"]["config_id"]

    listed = client.get("/agent/tool-configs?tool_key=geocoder&active_only=true")
    assert listed.status_code == 200
    assert listed.json()["items"]

    fetched = client.get(f"/agent/tool-configs/{config_id}")
    assert fetched.status_code == 200
    assert fetched.json()["item"]["tool_key"] == "geocoder"

    deleted = client.delete(f"/agent/tool-configs/{config_id}")
    assert deleted.status_code == 200

    missing = client.get(f"/agent/tool-configs/{config_id}")
    assert missing.status_code == 404
