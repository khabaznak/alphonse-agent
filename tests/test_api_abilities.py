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


def test_ability_crud_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_db(tmp_path, monkeypatch)
    client = TestClient(app)

    created = client.post(
        "/agent/abilities",
        json={
            "intent_name": "demo.echo",
            "kind": "tool_call_then_response",
            "tools": ["clock"],
            "enabled": True,
            "source": "user",
            "spec": {
                "responses": {
                    "default": "Echo ok",
                }
            },
        },
    )
    assert created.status_code == 201
    assert created.json()["item"]["intent_name"] == "demo.echo"

    fetched = client.get("/agent/abilities/demo.echo")
    assert fetched.status_code == 200
    assert fetched.json()["item"]["kind"] == "tool_call_then_response"

    patched = client.patch(
        "/agent/abilities/demo.echo",
        json={
            "enabled": False,
            "kind": "plan_emit",
            "spec": {
                "plan": {
                    "plan_type": "QUERY_STATUS",
                    "payload": {"include": ["gaps_summary"]},
                }
            },
        },
    )
    assert patched.status_code == 200
    assert patched.json()["item"]["enabled"] is False
    assert patched.json()["item"]["kind"] == "plan_emit"

    listed_all = client.get("/agent/abilities")
    assert listed_all.status_code == 200
    names = {item["intent_name"] for item in listed_all.json()["items"]}
    assert "demo.echo" in names

    listed_enabled = client.get("/agent/abilities?enabled_only=true")
    assert listed_enabled.status_code == 200
    enabled_names = {item["intent_name"] for item in listed_enabled.json()["items"]}
    assert "demo.echo" not in enabled_names

    deleted = client.delete("/agent/abilities/demo.echo")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True

    missing = client.get("/agent/abilities/demo.echo")
    assert missing.status_code == 404


def test_ability_create_rejects_mismatched_spec_intent_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    client = TestClient(app)
    response = client.post(
        "/agent/abilities",
        json={
            "intent_name": "demo.echo",
            "kind": "plan_emit",
            "tools": [],
            "spec": {"intent_name": "other.intent"},
        },
    )
    assert response.status_code == 400
