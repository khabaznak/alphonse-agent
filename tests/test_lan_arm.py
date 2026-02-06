from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from alphonse.infrastructure.api import app
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.lan.store import register_device


def test_lan_arm_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    register_device(device_id="device-1", device_name="Phone")

    client = TestClient(app)
    status = client.get("/lan/status").json()
    assert status["latest_device"]["armed"] is False

    arm = client.post("/lan/arm", json={}).json()
    assert arm["status"] == "armed"
    assert arm["device"]["armed"] is True

    status_after = client.get("/lan/status").json()
    assert status_after["latest_device"]["armed"] is True
