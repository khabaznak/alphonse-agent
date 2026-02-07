from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.cognition.intent_catalog import reset_catalog_service
from alphonse.agent.cortex.graph import invoke_cortex
from alphonse.agent.nervous_system.migrate import apply_schema


def _prepare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    reset_catalog_service()


def test_user_identity_sets_pending_interaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_db(tmp_path, monkeypatch)
    state = {
        "chat_id": "telegram:123",
        "channel_type": "telegram",
        "channel_target": "123",
        "timezone": "America/Mexico_City",
    }
    result = invoke_cortex(state, "Sabes como me llamo yo?", llm_client=None)
    assert result.meta.get("response_key") == "core.identity.user.ask_name"
    assert result.cognition_state.get("pending_interaction") is not None
