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


def test_onboarding_profile_api_crud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    client = TestClient(app)

    upsert = client.post(
        "/agent/onboarding/profiles",
        json={
            "principal_id": "person-alex",
            "state": "in_progress",
            "primary_role": "admin",
            "next_steps": ["home_location", "work_location"],
        },
    )
    assert upsert.status_code == 201
    assert upsert.json()["item"]["principal_id"] == "person-alex"

    listed = client.get("/agent/onboarding/profiles?state=in_progress")
    assert listed.status_code == 200
    assert listed.json()["items"]

    fetched = client.get("/agent/onboarding/profiles/person-alex")
    assert fetched.status_code == 200
    assert fetched.json()["item"]["primary_role"] == "admin"

    deleted = client.delete("/agent/onboarding/profiles/person-alex")
    assert deleted.status_code == 200

    missing = client.get("/agent/onboarding/profiles/person-alex")
    assert missing.status_code == 404


def test_location_and_device_location_api_crud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    client = TestClient(app)

    upsert = client.post(
        "/agent/locations",
        json={
            "principal_id": "person-alex",
            "label": "home",
            "address_text": "123 Main St, Guadalajara",
            "latitude": 20.67,
            "longitude": -103.35,
            "source": "user",
            "confidence": 0.9,
        },
    )
    assert upsert.status_code == 201
    item = upsert.json()["item"]
    location_id = item["location_id"]
    assert item["label"] == "home"

    listed = client.get("/agent/locations?principal_id=person-alex&active_only=true")
    assert listed.status_code == 200
    assert listed.json()["items"]

    fetched = client.get(f"/agent/locations/{location_id}")
    assert fetched.status_code == 200
    assert fetched.json()["item"]["address_text"] == "123 Main St, Guadalajara"

    device = client.post(
        "/agent/device-locations",
        json={
            "principal_id": "person-alex",
            "device_id": "pixel-001",
            "latitude": 20.68,
            "longitude": -103.34,
            "accuracy_meters": 8.0,
            "source": "alphonse_link",
            "metadata": {"battery": 85},
        },
    )
    assert device.status_code == 201
    assert device.json()["item"]["device_id"] == "pixel-001"

    device_list = client.get("/agent/device-locations?device_id=pixel-001")
    assert device_list.status_code == 200
    assert device_list.json()["items"]

    deleted = client.delete(f"/agent/locations/{location_id}")
    assert deleted.status_code == 200
