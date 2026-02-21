from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.senses.base import Sense, SignalSpec
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

logger = logging.getLogger(__name__)

MAX_ACCEPTABLE_TRIGGER_LATENCY_SECONDS = 30 * 60


@dataclass(frozen=True)
class TimedSignalRecord:
    id: str
    trigger_at: datetime
    timezone: str | None
    status: str
    signal_type: str
    payload: dict[str, Any]
    target: str | None
    origin: str | None
    correlation_id: str | None
    fired_at: datetime | None


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
            due_at = next_signal.trigger_at
            delta_seconds = (due_at - now).total_seconds()
            allowed_lag_seconds = _allowed_lag_seconds(next_signal, due_at)

            if delta_seconds < -allowed_lag_seconds:
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
                continue

            self._sleep(min(delta_seconds, self._poll_seconds))

    def _emit(self, record: TimedSignalRecord, due_at: datetime) -> None:
        if not self._bus:
            return
        mind_layer = str((record.payload or {}).get("mind_layer") or "subconscious").strip().lower()
        dispatch_mode = str((record.payload or {}).get("dispatch_mode") or "deterministic").strip().lower()
        self._bus.emit(
            Signal(
                type="timed_signal.fired",
                payload={
                    "timed_signal_id": record.id,
                    "mind_layer": mind_layer,
                    "dispatch_mode": dispatch_mode,
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
            "SELECT id, trigger_at, timezone, status, signal_type, payload, target, origin, correlation_id, fired_at "
            "FROM timed_signals WHERE status = 'pending' "
            "ORDER BY trigger_at ASC LIMIT 1"
        )
        with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
            row = conn.execute(query).fetchone()
        if not row:
            return None
        signal_id = str(row[0])
        try:
            trigger_at = _parse_dt(str(row[1]))
            fired_at = _parse_dt(str(row[9])) if row[9] else None
        except Exception:
            self._mark_error(signal_id, f"invalid_trigger_at:{row[1]}")
            logger.warning("TimerSense marked signal error id=%s invalid_trigger_at=%s", signal_id, row[1])
            return None
        timezone_value = _as_optional_str(row[2])
        payload = _parse_payload(row[5])
        return TimedSignalRecord(
            id=signal_id,
            trigger_at=trigger_at,
            timezone=timezone_value,
            status=str(row[3]),
            signal_type=str(row[4]),
            payload=payload,
            target=_as_optional_str(row[6]),
            origin=_as_optional_str(row[7]),
            correlation_id=_as_optional_str(row[8]),
            fired_at=fired_at,
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
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (signal_id,),
            )
            conn.commit()

    def _mark_error(self, signal_id: str, reason: str) -> None:
        _ = reason
        with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
            conn.execute(
                """
                UPDATE timed_signals
                SET status = 'error',
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (signal_id,),
            )
            conn.commit()

    def _sleep(self, seconds: float) -> None:
        self._stop_event.wait(timeout=max(0.0, seconds))


def _parse_dt(value: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise ValueError("empty datetime")
    normalized = text.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_payload(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_float(raw: str | None, default: float) -> float:
    try:
        value = float(str(raw).strip()) if raw is not None else float(default)
        if value <= 0:
            return float(default)
        return value
    except Exception:
        return float(default)


def _allowed_lag_seconds(record: TimedSignalRecord, due_at: datetime) -> float:
    _ = (record, due_at)
    return float(MAX_ACCEPTABLE_TRIGGER_LATENCY_SECONDS)
