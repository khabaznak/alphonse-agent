from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dateutil.rrule import rrulestr
from pathlib import Path
from typing import Any

from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimedSignalRecord:
    id: str
    trigger_at: datetime
    status: str
    signal_type: str
    payload: dict[str, Any]
    target: str | None
    origin: str | None
    correlation_id: str | None
    next_trigger_at: datetime | None
    rrule: str | None
    timezone: str | None


class TimedSignalScheduler:
    def __init__(
        self,
        *,
        db_path: str,
        bus: Bus,
        dispatch_window_seconds: int = 1800,
        idle_sleep_seconds: int = 60,
    ) -> None:
        self.db_path = str(db_path)
        self.bus = bus
        self.dispatch_window_seconds = dispatch_window_seconds
        self.idle_sleep_seconds = idle_sleep_seconds
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Timed signal scheduler started")

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Timed signal scheduler stopped")

    def notify_new_signal(self) -> None:
        self._wake_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                next_signal = self._fetch_next_pending()
            except Exception as exc:
                logger.warning("Timed scheduler query failed: %s", exc)
                self._sleep(self.idle_sleep_seconds)
                continue

            if next_signal is None:
                self._sleep(self.idle_sleep_seconds)
                continue

            now = datetime.now(tz=timezone.utc)
            due_at = next_signal.next_trigger_at or next_signal.trigger_at
            delta_seconds = (due_at - now).total_seconds()
            if delta_seconds < -self.dispatch_window_seconds:
                if next_signal.rrule:
                    next_at = _next_rrule_occurrence(next_signal, now)
                    if next_at:
                        self._update_next_trigger(next_signal.id, next_at)
                        continue
                self._mark_status(next_signal.id, "skipped")
                continue

            if delta_seconds <= self.dispatch_window_seconds:
                self._handle_dispatch(next_signal, due_at)
                continue

            wait_seconds = max(0.0, delta_seconds)
            self._sleep(min(wait_seconds, self.idle_sleep_seconds))

    def _emit(self, record: TimedSignalRecord, due_at: datetime) -> None:
        self.bus.emit(
            Signal(
                type="timed_signal_due",
                payload={
                    "timed_signal_id": record.id,
                    "signal_type": record.signal_type,
                    "payload": record.payload,
                    "target": record.target,
                    "origin": record.origin,
                    "correlation_id": record.correlation_id,
                    "trigger_at": due_at.isoformat(),
                },
                source="timed_scheduler",
            )
        )

    def _handle_dispatch(self, record: TimedSignalRecord, due_at: datetime) -> None:
        self._mark_status(record.id, "dispatched")
        if record.rrule:
            next_at = _next_rrule_occurrence(record, due_at)
            if next_at:
                self._update_next_trigger(record.id, next_at)
        self._emit(record, due_at)

    def _fetch_next_pending(self) -> TimedSignalRecord | None:
        query = (
            "SELECT id, trigger_at, next_trigger_at, rrule, timezone, status, signal_type, "
            "payload, target, origin, correlation_id "
            "FROM timed_signals WHERE status = 'pending' "
            "ORDER BY COALESCE(next_trigger_at, trigger_at) ASC LIMIT 1"
        )
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(query).fetchone()
        if not row:
            return None
        trigger_at = _parse_dt(row[1])
        next_trigger_at = _parse_dt(row[2]) if row[2] else None
        rrule_value = _as_optional_str(row[3])
        timezone_value = _as_optional_str(row[4])
        payload = _parse_payload(row[7])
        return TimedSignalRecord(
            id=str(row[0]),
            trigger_at=trigger_at,
            status=str(row[5]),
            signal_type=str(row[6]),
            payload=payload,
            target=_as_optional_str(row[8]),
            origin=_as_optional_str(row[9]),
            correlation_id=_as_optional_str(row[10]),
            next_trigger_at=next_trigger_at,
            rrule=rrule_value,
            timezone=timezone_value,
        )

    def _mark_status(self, signal_id: str, status: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE timed_signals SET status = ?, updated_at = datetime('now') WHERE id = ?",
                (status, signal_id),
            )
            conn.commit()

    def _update_next_trigger(self, signal_id: str, next_trigger_at: datetime) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE timed_signals SET next_trigger_at = ?, status = 'pending', updated_at = datetime('now') "
                "WHERE id = ?",
                (next_trigger_at.isoformat(), signal_id),
            )
            conn.commit()

    def _sleep(self, seconds: float) -> None:
        self._wake_event.wait(timeout=seconds)
        self._wake_event.clear()


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_payload(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _as_optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _next_rrule_occurrence(record: TimedSignalRecord, last_fire_at: datetime) -> datetime | None:
    tz_name = record.timezone or settings.get_timezone()
    try:
        tzinfo = ZoneInfo(tz_name)
    except Exception:
        tzinfo = timezone.utc

    dtstart = record.trigger_at.astimezone(tzinfo)
    rule = rrulestr(record.rrule or "", dtstart=dtstart)
    candidate = rule.after(last_fire_at.astimezone(tzinfo), inc=False)
    if candidate is None:
        return None
    if candidate.tzinfo is None:
        candidate = candidate.replace(tzinfo=tzinfo)
    return candidate.astimezone(timezone.utc)
