"""Runtime manager for background sense producers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from alphonse.agent.nervous_system.senses.base import Sense
from alphonse.agent.nervous_system.senses.bus import Bus
from alphonse.agent.nervous_system.senses.registry import all_senses


class SenseManager:
    def __init__(self, *, db_path: str, bus: Bus) -> None:
        self.db_path = db_path
        self.bus = bus
        self._senses: list[Sense] = []
        self._running = False

    def _connect(self) -> sqlite3.Connection:
        path = Path(self.db_path)
        return sqlite3.connect(path)

    def _enabled_sense_keys(self) -> set[str]:
        try:
            with self._connect() as conn:
                rows = conn.execute("SELECT key FROM senses WHERE enabled = 1").fetchall()
            return {row[0] for row in rows}
        except sqlite3.Error:
            return set()

    def start(self) -> None:
        if self._running:
            return
        enabled_keys = self._enabled_sense_keys()
        for sense_cls in all_senses():
            if sense_cls.key not in enabled_keys:
                continue
            sense = sense_cls()
            try:
                sense.start(self.bus)
            except Exception as exc:
                print(f"Sense failed to start ({sense_cls.key}): {exc}")
                continue
            self._senses.append(sense)
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        for sense in reversed(self._senses):
            try:
                sense.stop()
            except Exception as exc:
                print(f"Sense failed to stop ({sense.key}): {exc}")
        self._senses = []
        self._running = False
