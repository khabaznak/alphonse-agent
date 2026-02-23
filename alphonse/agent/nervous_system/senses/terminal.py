from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.agent.nervous_system.senses.base import Sense, SignalSpec
from alphonse.agent.nervous_system.senses.bus import Bus, Signal

logger = get_component_logger("senses.terminal")


class TerminalSense(Sense):
    key = "terminal"
    name = "Terminal Sense"
    description = "Emits signals when terminal commands change state."
    source_type = "system"
    signals = [
        SignalSpec(
            key="terminal.command_updated",
            name="Terminal Command Updated",
            description="Terminal command status change",
        )
    ]

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._bus: Bus | None = None
        self._poll_seconds = _parse_float(
            os.getenv("TERMINAL_SENSE_POLL_SECONDS"),
            5.0,
        )
        self._enabled = os.getenv("TERMINAL_SENSE_ENABLED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._last_seen = _now_iso()

    def start(self, bus: Bus) -> None:
        if not self._enabled:
            logger.info("TerminalSense disabled")
            return
        if self._thread and self._thread.is_alive():
            return
        self._bus = bus
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("TerminalSense started interval=%.2fs", self._poll_seconds)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("TerminalSense stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            updates = _fetch_updates(self._last_seen)
            for row in updates:
                self._last_seen = max(self._last_seen, row["updated_at"])
                if not self._bus:
                    continue
                self._bus.emit(
                    Signal(
                        type="terminal.command_updated",
                        payload=row,
                        source="terminal",
                        correlation_id=row.get("command_id"),
                    )
                )
            self._stop_event.wait(timeout=self._poll_seconds)


def _fetch_updates(since_iso: str) -> list[dict[str, object]]:
    db_path = resolve_nervous_system_db_path()
    if not Path(db_path).exists():
        return []
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT command_id, session_id, command, cwd, status, stdout, stderr, exit_code,
                       requested_by, approved_by, created_at, updated_at
                FROM terminal_commands
                WHERE updated_at > ?
                ORDER BY updated_at ASC
                """,
                (since_iso,),
            ).fetchall()
    except sqlite3.Error:
        return []
    updates: list[dict[str, object]] = []
    for row in rows:
        updates.append(
            {
                "command_id": row[0],
                "session_id": row[1],
                "command": row[2],
                "cwd": row[3],
                "status": row[4],
                "stdout": row[5],
                "stderr": row[6],
                "exit_code": row[7],
                "requested_by": row[8],
                "approved_by": row[9],
                "created_at": row[10],
                "updated_at": row[11],
            }
        )
    return updates


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return max(1.0, float(raw))
    except ValueError:
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
