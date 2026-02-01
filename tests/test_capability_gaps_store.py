from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.nervous_system.capability_gaps import insert_gap, get_gap
from alphonse.agent.nervous_system.migrate import apply_schema


def test_insert_gap_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    gap_id = insert_gap(
        {
            "user_text": "Need a calendar sync",
            "reason": "unknown_intent",
            "status": "open",
            "intent": "unknown",
            "confidence": 0.2,
            "missing_slots": ["calendar"],
            "channel_type": "cli",
            "channel_id": "cli",
            "correlation_id": "test-1",
        }
    )

    row = get_gap(gap_id)
    assert row is not None
    assert row.get("gap_id") == gap_id
    assert row.get("reason") == "unknown_intent"
    assert row.get("status") == "open"
