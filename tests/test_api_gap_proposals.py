from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from alphonse.agent.nervous_system.capability_gaps import insert_gap
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.infrastructure.api import app


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)


def test_gap_proposal_crud_and_dispatch_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    client = TestClient(app)

    create = client.post(
        "/agent/gap-proposals",
        json={
            "gap_id": "gap-001",
            "proposed_category": "intent_missing",
            "proposed_next_action": "plan",
            "proposed_intent_name": "time",
            "confidence": 0.9,
            "notes": '{"source":"manual"}',
            "language": "es",
        },
    )
    assert create.status_code == 201
    proposal_id = create.json()["id"]

    fetched = client.get(f"/agent/gap-proposals/{proposal_id}")
    assert fetched.status_code == 200
    assert fetched.json()["item"]["proposed_intent_name"] == "time"

    updated = client.patch(
        f"/agent/gap-proposals/{proposal_id}",
        json={"status": "approved", "reviewer": "alex", "notes": "approve time intent"},
    )
    assert updated.status_code == 200
    assert updated.json()["item"]["status"] == "approved"

    dispatched = client.post(
        f"/agent/gap-proposals/{proposal_id}/dispatch",
        json={"task_type": "plan", "actor": "alex"},
    )
    assert dispatched.status_code == 200
    task_id = dispatched.json()["task_id"]
    assert dispatched.json()["task"]["type"] == "plan"

    task_read = client.get(f"/agent/gap-tasks/{task_id}")
    assert task_read.status_code == 200
    assert task_read.json()["item"]["status"] == "open"

    task_done = client.patch(f"/agent/gap-tasks/{task_id}", json={"status": "done"})
    assert task_done.status_code == 200
    assert task_done.json()["item"]["status"] == "done"

    deleted = client.delete(f"/agent/gap-proposals/{proposal_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True


def test_gap_proposal_coalesce_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    insert_gap(
        {
            "user_text": "What time is it?",
            "reason": "proposed_intent_unmapped",
            "status": "open",
            "metadata": {
                "proposed_intent": "time",
                "proposed_intent_aliases": ["current time"],
                "proposed_intent_confidence": 0.92,
            },
        }
    )
    insert_gap(
        {
            "user_text": "Que horas son?",
            "reason": "proposed_intent_unmapped",
            "status": "open",
            "metadata": {
                "proposed_intent": "time",
                "proposed_intent_aliases": ["hora"],
                "proposed_intent_confidence": 0.9,
            },
        }
    )
    client = TestClient(app)
    response = client.post(
        "/agent/gap-proposals/coalesce",
        json={"limit": 100, "min_cluster_size": 2},
    )
    assert response.status_code == 200
    assert response.json()["created_count"] == 1

    listed = client.get("/agent/gap-proposals?status=pending&limit=10")
    assert listed.status_code == 200
    items = listed.json()["items"]
    assert items
    assert items[0]["proposed_intent_name"] == "time"
