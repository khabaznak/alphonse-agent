from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.agent.services.job_runner import JobRunner
from alphonse.agent.services.job_store import JobStore, compute_next_run_at


class ScheduledJobsReconciler:
    """Deterministic reconciler that keeps scheduled_jobs and timed_signals aligned."""

    def __init__(self, *, store: JobStore | None = None) -> None:
        self._store = store or JobStore()

    def reconcile(
        self,
        *,
        now: datetime | None = None,
        brain_event_sink: Any | None = None,
    ) -> dict[str, Any]:
        current = now or datetime.now(timezone.utc)
        summary = self._store.backfill_and_sync_jobs()
        stale_removed = self._remove_stale_scheduled_jobs()
        catchup = self._catch_up_due_jobs(current=current, brain_event_sink=brain_event_sink)
        self._cleanup_legacy_daily_report_signal()
        return {
            "scanned": int(summary.get("scanned") or 0),
            "updated": int(summary.get("updated") or 0),
            "stale_removed": stale_removed,
            "executed": int(catchup.get("executed") or 0),
            "advanced_without_run": int(catchup.get("advanced_without_run") or 0),
            "failed": int(catchup.get("failed") or 0),
            "overdue_active_jobs": self._count_overdue_active_jobs(current),
            "due_pending_timed_signals": self._count_due_pending_timed_signals(),
        }

    def _catch_up_due_jobs(self, *, current: datetime, brain_event_sink: Any | None) -> dict[str, int]:
        grace_seconds = _parse_int_env("JOB_RECONCILIATION_CATCHUP_GRACE_SECONDS", 6 * 60 * 60)
        max_jobs = _parse_int_env("JOB_RECONCILIATION_MAX_JOBS_PER_TICK", 25)
        runner = JobRunner(
            job_store=self._store,
            tick_seconds=45,
        )
        executed = 0
        advanced_without_run = 0
        failed = 0
        processed = 0
        for user_id in self._store.list_user_ids():
            due = self._store.due_jobs(user_id=user_id, now=current)
            for job in due:
                if processed >= max_jobs:
                    return {
                        "executed": executed,
                        "advanced_without_run": advanced_without_run,
                        "failed": failed,
                    }
                processed += 1
                overdue_seconds = _overdue_seconds(job.next_run_at, now=current)
                if overdue_seconds is None:
                    continue
                if overdue_seconds <= grace_seconds:
                    outcome = runner.run_job_now(
                        user_id=user_id,
                        job_id=job.job_id,
                        now=current,
                        brain_event_sink=brain_event_sink if callable(brain_event_sink) else None,
                    )
                    if str(outcome.get("status") or "").strip().lower() == "error":
                        failed += 1
                    else:
                        executed += 1
                    continue
                # Too old: advance deterministically without replaying stale runs.
                job.last_run_at = current.isoformat()
                job.next_run_at = compute_next_run_at(
                    schedule=job.schedule,
                    timezone_name=job.timezone,
                    after=current,
                )
                self._store.save_job(user_id=user_id, job=job)
                advanced_without_run += 1
        return {
            "executed": executed,
            "advanced_without_run": advanced_without_run,
            "failed": failed,
        }

    def _remove_stale_scheduled_jobs(self) -> int:
        removed = 0
        user_ids = set(self._store.list_user_ids())
        with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, owner_id FROM scheduled_jobs ORDER BY updated_at DESC"
            ).fetchall()
            for row in rows:
                job_id = str(row["id"] or "").strip()
                owner_id = str(row["owner_id"] or "").strip()
                if not job_id or not owner_id:
                    continue
                if owner_id not in user_ids:
                    conn.execute("DELETE FROM scheduled_jobs WHERE id = ?", (job_id,))
                    conn.execute("DELETE FROM timed_signals WHERE id = ?", (f"job_trigger:{job_id}",))
                    removed += 1
                    continue
                try:
                    self._store.get_job(user_id=owner_id, job_id=job_id)
                except Exception:
                    conn.execute("DELETE FROM scheduled_jobs WHERE id = ?", (job_id,))
                    conn.execute("DELETE FROM timed_signals WHERE id = ?", (f"job_trigger:{job_id}",))
                    removed += 1
            conn.commit()
        return removed

    def _cleanup_legacy_daily_report_signal(self) -> None:
        with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
            conn.execute(
                """
                DELETE FROM timed_signals
                WHERE id = 'daily_report'
                  AND signal_type = 'timed_signal'
                """
            )
            conn.commit()

    def _count_overdue_active_jobs(self, now: datetime) -> int:
        with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM scheduled_jobs
                WHERE status = 'active'
                  AND next_run_at IS NOT NULL
                  AND datetime(next_run_at) <= datetime(?)
                """,
                (now.isoformat(),),
            ).fetchone()
        return int(row[0]) if row else 0

    def _count_due_pending_timed_signals(self) -> int:
        with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM timed_signals
                WHERE status = 'pending'
                  AND datetime(trigger_at) <= datetime('now')
                """
            ).fetchone()
        return int(row[0]) if row else 0


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        value = int(str(raw).strip())
    except Exception:
        return int(default)
    if value <= 0:
        return int(default)
    return value


def _overdue_seconds(next_run_at: str | None, *, now: datetime) -> int | None:
    value = str(next_run_at or "").strip()
    if not value:
        return None
    try:
        due = datetime.fromisoformat(value)
    except Exception:
        return None
    if due.tzinfo is None:
        due = due.replace(tzinfo=timezone.utc)
    return max(int((now - due.astimezone(timezone.utc)).total_seconds()), 0)
