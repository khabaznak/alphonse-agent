from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.pdca_queue_store import append_pdca_event, get_pdca_queue_metrics, upsert_pdca_task


def test_pdca_queue_metrics_first_cut_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    now = datetime.now(timezone.utc)
    task_a1 = upsert_pdca_task(
        {
            "owner_id": "owner-a",
            "conversation_key": "chat-a1",
            "status": "queued",
            "next_run_at": (now - timedelta(seconds=120)).isoformat(),
            "priority": 100,
        }
    )
    task_a2 = upsert_pdca_task(
        {
            "owner_id": "owner-a",
            "conversation_key": "chat-a2",
            "status": "running",
            "next_run_at": (now - timedelta(seconds=30)).isoformat(),
            "priority": 100,
        }
    )
    task_b1 = upsert_pdca_task(
        {
            "owner_id": "owner-b",
            "conversation_key": "chat-b1",
            "status": "queued",
            "next_run_at": (now - timedelta(seconds=60)).isoformat(),
            "priority": 100,
        }
    )
    _ = upsert_pdca_task(
        {
            "owner_id": "owner-c",
            "conversation_key": "chat-c1",
            "status": "done",
            "priority": 100,
        }
    )

    _ = append_pdca_event(task_id=task_a1, event_type="slice.requested")
    _ = append_pdca_event(task_id=task_a1, event_type="slice.requested")
    _ = append_pdca_event(task_id=task_b1, event_type="slice.requested")
    _ = append_pdca_event(
        task_id=task_a2,
        event_type="slice.blocked.budget_exhausted",
        payload={"reason": "max_cycles_reached"},
    )
    _ = append_pdca_event(
        task_id=task_b1,
        event_type="slice.blocked.budget_exhausted",
        payload={"reason": "token_budget_exhausted"},
    )
    _ = append_pdca_event(task_id=task_a1, event_type="queue.starvation_warning")
    _ = append_pdca_event(task_id=task_a1, event_type="slice.completed.done")
    _ = append_pdca_event(task_id=task_b1, event_type="slice.blocked.missing_text")
    _ = append_pdca_event(task_id=task_a2, event_type="slice.failed")

    metrics = get_pdca_queue_metrics(now=now.isoformat(), lookback_minutes=15)

    assert metrics["queue_depth_total"] == 4
    assert metrics["queue_depth_by_status"]["queued"] == 2
    assert metrics["queue_depth_by_status"]["running"] == 1
    assert metrics["queue_depth_by_status"]["done"] == 1
    assert metrics["queue_depth_by_owner"]["owner-a"] == 2
    assert metrics["queue_depth_by_owner"]["owner-b"] == 1

    assert int(metrics["oldest_wait_seconds"]["global"]) >= 120
    assert int(metrics["oldest_wait_seconds"]["by_owner"]["owner-a"]) >= 120
    assert int(metrics["oldest_wait_seconds"]["by_owner"]["owner-b"]) >= 60

    assert float(metrics["dispatch_rate_per_minute"]["global"]) == pytest.approx(0.2, rel=1e-3)
    assert float(metrics["dispatch_rate_per_minute"]["by_owner"]["owner-a"]) == pytest.approx(2 / 15, rel=1e-3)
    assert float(metrics["dispatch_rate_per_minute"]["by_owner"]["owner-b"]) == pytest.approx(1 / 15, rel=1e-3)

    assert metrics["budget_exhaustions_total"]["all"] == 2
    assert metrics["budget_exhaustions_total"]["by_reason"]["max_cycles_reached"] == 1
    assert metrics["budget_exhaustions_total"]["by_reason"]["token_budget_exhausted"] == 1
    assert metrics["starvation_warnings_total"] == 1

    fairness = metrics["owner_fairness_ratio"]["by_owner"]
    assert fairness["owner-a"] == pytest.approx(1.0, rel=1e-3)
    assert fairness["owner-b"] == pytest.approx(1.0, rel=1e-3)

    assert metrics["terminal_outcomes_total"]["done"] == 1
    assert metrics["terminal_outcomes_total"]["waiting_user"] == 1
    assert metrics["terminal_outcomes_total"]["failed"] == 3
