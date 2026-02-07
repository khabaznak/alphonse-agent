from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from dateutil.rrule import rrulestr

from alphonse.agent.nervous_system.senses.base import Sense, SignalSpec
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
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


class TimerSense(Sense):
    key = "timer"
    name = "Timer Sense"
    description = "Emits timed_signal.fired signals when timed signals are due"
    source_type = "system"
    signals = [
        SignalSpec(key="timed_signal.fired", name="Timed Signal Fired", description="Timed signal fired"),
    ]

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._bus: Bus | None = None
        self._poll_seconds = _parse_float(os.getenv("TIMER_POLL_SECONDS"), 60.0)

    def start(self, bus: Bus) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._bus = bus
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("TimerSense started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("TimerSense stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            next_signal = self._fetch_next_pending()
            if next_signal is None:
                logger.info("TimerSense tick status=idle")
                self._sleep(self._poll_seconds)
                continue

            now = datetime.now(tz=timezone.utc)
            due_at = next_signal.next_trigger_at or next_signal.trigger_at
            delta_seconds = (due_at - now).total_seconds()

            if delta_seconds < -1800:
                if next_signal.rrule:
                    next_at = _next_rrule_occurrence(next_signal, now)
                    if next_at:
                        self._reschedule(next_signal.id, next_at)
                        continue
                self._mark_failed(next_signal.id, "missed dispatch window")
                continue

            if delta_seconds <= 0:
                fired = self._mark_fired(next_signal.id)
                if fired:
                    logger.info(
                        "TimerSense firing id=%s signal_type=%s correlation_id=%s due_at=%s now_utc=%s lag_seconds=%.3f",
                        next_signal.id,
                        next_signal.signal_type,
                        next_signal.correlation_id,
                        due_at.isoformat(),
                        now.isoformat(),
                        (now - due_at).total_seconds(),
                    )
                    self._emit(next_signal, due_at)
                    if next_signal.rrule:
                        next_at = _next_rrule_occurrence(next_signal, due_at)
                        if next_at:
                            self._reschedule(next_signal.id, next_at)
                continue

            self._sleep(min(delta_seconds, self._poll_seconds))

    def _emit(self, record: TimedSignalRecord, due_at: datetime) -> None:
        if not self._bus:
            return
        self._bus.emit(
            Signal(
                type="timed_signal.fired",
                payload={
                    "timed_signal_id": record.id,
                    "signal_type": record.signal_type,
                    "payload": record.payload,
                    "target": record.target,
                    "origin": record.origin,
                    "correlation_id": record.correlation_id,
                    "trigger_at": due_at.isoformat(),
                },
                source="timer",
                correlation_id=record.correlation_id,
            )
        )

    def _fetch_next_pending(self) -> TimedSignalRecord | None:
        query = (
            "SELECT id, trigger_at, next_trigger_at, rrule, timezone, status, signal_type, "
            "payload, target, origin, correlation_id "
            "FROM timed_signals WHERE status = 'pending' "
            "ORDER BY COALESCE(next_trigger_at, trigger_at) ASC LIMIT 1"
        )
        with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
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
        with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
            conn.execute(
                "UPDATE timed_signals SET status = ?, updated_at = datetime('now') WHERE id = ?",
                (status, signal_id),
            )
            conn.commit()

    def _mark_fired(self, signal_id: str) -> bool:
        with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
            cur = conn.execute(
                """
                UPDATE timed_signals
                SET status = 'fired',
                    fired_at = datetime('now'),
                    attempt_count = attempt_count + 1,
                    attempts = attempts + 1,
                    last_error = NULL,
                    updated_at = datetime('now')
                WHERE id = ? AND status = 'pending'
                """,
                (signal_id,),
            )
            conn.commit()
            return cur.rowcount == 1

    def _mark_failed(self, signal_id: str, reason: str) -> None:
        with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
            conn.execute(
                """
                UPDATE timed_signals
                SET status = 'failed',
                    last_error = ?,
                    attempt_count = attempt_count + 1,
                    attempts = attempts + 1,
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (reason, signal_id),
            )
            conn.commit()

    def _reschedule(self, signal_id: str, next_trigger_at: datetime) -> None:
        with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
            conn.execute(
                "UPDATE timed_signals SET next_trigger_at = ?, status = 'pending', updated_at = datetime('now') "
                "WHERE id = ?",
                (next_trigger_at.isoformat(), signal_id),
            )
            conn.commit()

    def _sleep(self, seconds: float) -> None:
        self._stop_event.wait(timeout=seconds)


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
    dtstart = record.trigger_at
    rule = rrulestr(record.rrule or "", dtstart=dtstart)
    candidate = rule.after(last_fire_at, inc=False)
    return candidate.astimezone(timezone.utc) if candidate else None


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
