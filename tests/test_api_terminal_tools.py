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


def test_terminal_api_crud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    client = TestClient(app)

    sandbox = client.post(
        "/agent/terminal/sandboxes",
        json={
            "owner_principal_id": "principal-123",
            "label": "Projects",
            "path": "/tmp/projects",
            "is_active": True,
        },
    )
    assert sandbox.status_code == 201
    sandbox_id = sandbox.json()["item"]["sandbox_id"]

    listed = client.get(
        "/agent/terminal/sandboxes?owner_principal_id=principal-123&active_only=true"
    )
    assert listed.status_code == 200
    assert listed.json()["items"]

    created = client.post(
        "/agent/terminal/commands",
        json={
            "principal_id": "principal-123",
            "sandbox_id": sandbox_id,
            "command": "ls -la",
            "cwd": ".",
            "requested_by": "principal-123",
        },
    )
    assert created.status_code == 201
    command_id = created.json()["item"]["command_id"]

    approved = client.post(
        f"/agent/terminal/commands/{command_id}/approve",
        json={"approved_by": "principal-123"},
    )
    assert approved.status_code == 200
    assert approved.json()["item"]["status"] == "approved"

    finalized = client.post(
        f"/agent/terminal/commands/{command_id}/finalize",
        json={
            "stdout": "file1.txt\n",
            "stderr": "",
            "exit_code": 0,
            "status": "executed",
        },
    )
    assert finalized.status_code == 200
    assert finalized.json()["item"]["status"] == "executed"

    deleted = client.delete(f"/agent/terminal/sandboxes/{sandbox_id}")
    assert deleted.status_code == 200

    missing = client.get(f"/agent/terminal/sandboxes/{sandbox_id}")
    assert missing.status_code == 404
