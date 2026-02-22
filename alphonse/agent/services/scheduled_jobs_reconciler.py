from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.agent.services.job_store import JobStore


class ScheduledJobsReconciler:
    """Deterministic reconciler that keeps scheduled_jobs and timed_signals aligned."""

    def __init__(self, *, store: JobStore | None = None) -> None:
        self._store = store or JobStore()

    def reconcile(self, *, now: datetime | None = None) -> dict[str, Any]:
        current = now or datetime.now(timezone.utc)
        summary = self._store.backfill_and_sync_jobs()
        return {
            "scanned": int(summary.get("scanned") or 0),
            "updated": int(summary.get("updated") or 0),
            "overdue_active_jobs": self._count_overdue_active_jobs(current),
            "due_pending_timed_signals": self._count_due_pending_timed_signals(),
        }

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
