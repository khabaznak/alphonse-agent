from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.brain.habits_db import get_plan_run_by_correlation, list_enabled_habits_for_trigger
from alphonse.brain.orchestrator import handle_event


def test_habit_crystallization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    import alphonse.brain.graphs.executor_graph as executor_graph

    monkeypatch.setattr(executor_graph, "dispatch", lambda skill, args, context: ("sent", {}))

    context = {"severity": "critical", "requires_ack": True, "ttl_sec": 300}
    payload = {
        "pairing_id": "pair-1",
        "device_name": "Test Phone",
        "otp": "OTP",
        "expires_at": "2026-02-05T00:00:00+00:00",
    }

    result = handle_event("pairing.requested", context, payload)
    assert result["status"] == "planned"

    habits = list_enabled_habits_for_trigger("pairing.requested")
    assert len(habits) == 1

    run = get_plan_run_by_correlation("pair-1")
    assert run is not None
    assert run.habit_id is None

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM delivery_receipts").fetchone()[0]
        assert count == 2

    payload2 = dict(payload)
    payload2["pairing_id"] = "pair-2"
    result2 = handle_event("pairing.requested", context, payload2)
    assert result2["status"] == "habit_executed"

    run2 = get_plan_run_by_correlation("pair-2")
    assert run2 is not None
    assert run2.habit_id == habits[0].habit_id
