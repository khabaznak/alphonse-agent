from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from alphonse.agent.cognition.status_summaries import summarize_timed_signals
from alphonse.agent.nervous_system.migrate import apply_schema


def test_timed_signals_summary_with_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    payload = json.dumps({"message": "Drink water"})
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO timed_signals (id, trigger_at, signal_type, payload, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("ts-1", "2026-02-01T10:00:00Z", "reminder", payload, "pending"),
        )
        conn.commit()

    summary = summarize_timed_signals("en-US", limit=10)
    assert "Upcoming reminders" in summary
    assert "Drink water" in summary
