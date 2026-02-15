from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from alphonse.agent.nervous_system.assets import get_asset, resolve_asset_path
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.infrastructure.api import app


def test_agent_assets_upload_creates_asset_record(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    sandbox_root = tmp_path / "sandbox"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHONSE_SANDBOX_ROOT", str(sandbox_root))
    apply_schema(db_path)

    client = TestClient(app)
    response = client.post(
        "/agent/assets",
        data={
            "user_id": "web-user-1",
            "provider": "webui",
            "channel": "webui",
            "target": "webui",
            "kind": "audio",
        },
        files={"file": ("voice.wav", b"FAKE-WAVE", "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("asset_id"), str)
    assert payload.get("mime") == "audio/wav"
    assert payload.get("bytes") == len(b"FAKE-WAVE")

    stored = get_asset(payload["asset_id"])
    assert isinstance(stored, dict)
    assert stored.get("kind") == "audio"
    assert stored.get("mime") == "audio/wav"
    resolved = resolve_asset_path(payload["asset_id"])
    assert resolved.exists()
    assert resolved.read_bytes() == b"FAKE-WAVE"
