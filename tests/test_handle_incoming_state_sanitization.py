from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphonse.agent.actions.handle_incoming_message import (
    IncomingContext,
    _build_cortex_state,
)
from alphonse.agent.nervous_system.migrate import apply_schema


def _setup_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)


def test_build_cortex_state_clears_expired_pending_interaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_db(tmp_path, monkeypatch)
    expired = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    stored_state = {
        "pending_interaction": {
            "type": "SLOT_FILL",
            "key": "answer",
            "context": {"tool": "askQuestion"},
            "created_at": "now",
            "expires_at": expired,
        },
        "ability_state": {"kind": "discovery_loop", "steps": [{"tool": "askQuestion"}]},
    }
    incoming = IncomingContext(
        channel_type="telegram",
        address="123",
        person_id=None,
        correlation_id="corr-1",
        update_id=None,
    )

    state = _build_cortex_state(
        stored_state,
        incoming,
        "corr-1",
        {},
        SimpleNamespace(user_id=None, user_name=None, metadata={}),
    )

    assert state.get("pending_interaction") is None
    assert state.get("ability_state") == stored_state["ability_state"]


def test_build_cortex_state_keeps_active_pending_interaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_db(tmp_path, monkeypatch)
    active = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    pending = {
        "type": "SLOT_FILL",
        "key": "answer",
        "context": {"tool": "askQuestion"},
        "created_at": "now",
        "expires_at": active,
    }
    ability_state = {"kind": "discovery_loop", "steps": [{"tool": "askQuestion"}]}
    stored_state = {
        "pending_interaction": pending,
        "ability_state": ability_state,
    }
    incoming = IncomingContext(
        channel_type="telegram",
        address="123",
        person_id=None,
        correlation_id="corr-1",
        update_id=None,
    )

    state = _build_cortex_state(
        stored_state,
        incoming,
        "corr-1",
        {},
        SimpleNamespace(user_id=None, user_name=None, metadata={}),
    )

    assert state.get("pending_interaction") == pending
    assert state.get("ability_state") == ability_state
