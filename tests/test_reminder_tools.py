from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.tools.reminder_tools import ReminderCancelTool
from alphonse.agent.tools.reminder_tools import ReminderListTool


def _insert_timed_signal(
    db_path: Path,
    *,
    signal_id: str,
    trigger_at: str,
    status: str = "pending",
    payload: dict[str, object] | None = None,
    target: str = "8553589429",
    origin: str = "telegram",
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO timed_signals
              (id, trigger_at, timezone, status, fired_at, signal_type, payload, target, origin, correlation_id)
            VALUES
              (?, ?, ?, ?, NULL, 'timed_signal', ?, ?, ?, ?)
            """,
            (
                signal_id,
                trigger_at,
                "America/Mexico_City",
                status,
                json.dumps(payload or {}),
                target,
                origin,
                f"corr-{signal_id}",
            ),
        )
        conn.commit()


def test_reminders_list_returns_pending_timed_signal_reminders_not_jobs(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    _insert_timed_signal(
        db_path,
        signal_id="reminder_7pm",
        trigger_at="2026-05-22T01:00:00+00:00",
        payload={
            "kind": "reminder",
            "user_id": "u-alex",
            "reminder_text_raw": "Recordatorio de reunion en casa de primo Lalo.",
            "origin_channel": "telegram",
        },
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO scheduled_jobs (id, name, owner_id, status, next_run_at, timezone)
            VALUES ('job_7pm', 'unrelated job', 'u-alex', 'active', '2026-05-22T01:00:00+00:00', 'America/Mexico_City')
            """
        )
        conn.commit()

    listed = ReminderListTool().execute(
        owner_id="u-alex",
        start_at="2026-05-21T00:00:00-06:00",
        end_at="2026-05-21T23:59:59-06:00",
        text="reunion",
    )

    assert listed["exception"] is None
    rows = listed["output"]["reminders"]
    assert [row["reminder_id"] for row in rows] == ["reminder_7pm"]
    assert rows[0]["reminder_text_raw"] == "Recordatorio de reunion en casa de primo Lalo."


def test_reminders_cancel_marks_pending_row_cancelled(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    _insert_timed_signal(
        db_path,
        signal_id="reminder_pending",
        trigger_at="2026-05-22T01:00:00+00:00",
        payload={"kind": "reminder", "user_id": "u-alex", "reminder_text_raw": "Meeting"},
    )

    cancelled = ReminderCancelTool().execute(reminder_id="reminder_pending")

    assert cancelled["exception"] is None
    assert cancelled["output"]["cancelled"] is True
    with sqlite3.connect(db_path) as conn:
        status = conn.execute("SELECT status FROM timed_signals WHERE id = 'reminder_pending'").fetchone()[0]
    assert status == "cancelled"


def test_reminders_cancel_refuses_fired_reminder(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    _insert_timed_signal(
        db_path,
        signal_id="reminder_fired",
        trigger_at="2026-05-22T01:00:00+00:00",
        status="fired",
        payload={"kind": "reminder", "user_id": "u-alex", "reminder_text_raw": "Meeting"},
    )

    cancelled = ReminderCancelTool().execute(reminder_id="reminder_fired")

    assert cancelled["output"] is None
    assert cancelled["exception"]["code"] == "reminder_already_fired"
    with sqlite3.connect(db_path) as conn:
        status = conn.execute("SELECT status FROM timed_signals WHERE id = 'reminder_fired'").fetchone()[0]
    assert status == "fired"


def test_reminders_cancel_can_use_single_high_confidence_match(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    _insert_timed_signal(
        db_path,
        signal_id="reminder_match",
        trigger_at="2026-05-22T01:00:00+00:00",
        payload={"kind": "reminder", "user_id": "u-alex", "reminder_text_raw": "Meeting with cousin"},
    )

    cancelled = ReminderCancelTool().execute(
        owner_id="u-alex",
        start_at="2026-05-21T18:30:00-06:00",
        end_at="2026-05-21T19:30:00-06:00",
        text="cousin",
    )

    assert cancelled["exception"] is None
    assert cancelled["output"]["reminder_id"] == "reminder_match"
