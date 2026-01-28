"""Data-driven finite state machine (FSM) backed by SQLite."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time
import threading

from rex.senses.bus import Signal as BusSignal


@dataclass
class DDFSMConfig:
    db_path: str
    queries_path: str | None = None

    # Cache controls
    cache_enabled: bool = True
    cache_ttl_seconds: float = 30.0





@dataclass
class CurrentState:
    id: int
    key: str | None = None
    name: str | None = None


@dataclass
class TransitionOutcome:
    """Result of attempting to apply a signal while in a given state."""

    matched: bool
    reason: str

    # Transition metadata (optional but useful for auditing)
    transition_id: int | None = None
    guard_key: str | None = None
    action_key: str | None = None

    # Next state fields (preferred)
    next_state_id: int | None = None
    next_state_key: str | None = None
    next_state_name: str | None = None

    # Backward-compat convenience
    id: int | None = None
    key: str | None = None
    name: str | None = None


class DDFSM:
    def __init__(self, config: DDFSMConfig) -> None:
        self.config = config
        self._queries = self._load_queries()
        self._lock = threading.Lock()
        self._cache: dict[tuple[int, str], tuple[float, TransitionOutcome]] = {}

    def _load_queries(self) -> dict[str, str]:
        if self.config.queries_path:
            path = Path(self.config.queries_path)
        else:
            path = Path(__file__).resolve().parent / "resources" / "queries.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.config.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def execute(self, query_key: str, params: dict[str, object]) -> list[sqlite3.Row]:
        sql = self._queries.get(query_key)
        if not sql:
            raise KeyError(f"Query not found: {query_key}")
        with self._connect() as conn:
            cur = conn.execute(sql, params)
            return cur.fetchall()

    def _cache_get(self, state_id: int, signal_key: str) -> Optional[TransitionOutcome]:
        if not self.config.cache_enabled:
            return None
        now = time.time()
        key = (state_id, signal_key)
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            expires_at, outcome = entry
            if expires_at < now:
                self._cache.pop(key, None)
                return None
            return outcome

    def _cache_set(self, state_id: int, signal_key: str, outcome: TransitionOutcome) -> None:
        if not self.config.cache_enabled:
            return
        key = (state_id, signal_key)
        expires_at = time.time() + float(self.config.cache_ttl_seconds)
        with self._lock:
            self._cache[key] = (expires_at, outcome)

    def invalidate_cache(self) -> None:
        """Clear any cached transition lookups."""
        with self._lock:
            self._cache.clear()

    def get_transition(self, state_id: int, signal_id: int) -> sqlite3.Row | None:
        rows = self.execute(
            "transition_lookup",
            {"state_id": state_id, "signal_id": signal_id},
        )
        return rows[0] if rows else None

    def lookup_outcome(self, state_id: int, signal_key: str) -> TransitionOutcome:
        """Lookup transition outcome for (state_id, signal_key).

        Uses a query that should return the next state's id/key/name plus optional
        transition metadata such as guard/action keys.
        """
        rows = self.execute(
            "transition_next_state_by_signal_key",
            {"state_id": state_id, "signal_key": signal_key},
        )
        if not rows:
            return TransitionOutcome(
                matched=False,
                reason="NO_TRANSITION",
            )

        row = rows[0]

        # Required next-state fields (names depend on your queries.json aliases)
        next_id = row.get("next_state_id") if hasattr(row, "get") else row["next_state_id"] if "next_state_id" in row.keys() else row["state_id"]
        next_key = row.get("next_state_key") if hasattr(row, "get") else row["next_state_key"] if "next_state_key" in row.keys() else row["state_key"]
        next_name = row.get("next_state_name") if hasattr(row, "get") else row["next_state_name"] if "next_state_name" in row.keys() else row["state_name"]

        outcome = TransitionOutcome(
            matched=True,
            reason="MATCH",
            transition_id=row["transition_id"] if "transition_id" in row.keys() else None,
            guard_key=row["guard_key"] if "guard_key" in row.keys() else None,
            action_key=row["action_key"] if "action_key" in row.keys() else None,
            next_state_id=int(next_id) if next_id is not None else None,
            next_state_key=str(next_key) if next_key is not None else None,
            next_state_name=str(next_name) if next_name is not None else None,
            # backward-compat
            id=int(next_id) if next_id is not None else None,
            key=str(next_key) if next_key is not None else None,
            name=str(next_name) if next_name is not None else None,
        )
        return outcome

    def handle(self, state: CurrentState, signal: BusSignal, ctx: object) -> TransitionOutcome:
        """Apply a signal to the current state and return the transition outcome."""
        _ = ctx  # reserved for future policy/guard evaluation.

        # If we ever introduce a reload signal, allow it to invalidate cache.
        if signal.type == "RELOAD_FSM":
            self.invalidate_cache()
            return TransitionOutcome(matched=False, reason="CACHE_INVALIDATED")

        cached = self._cache_get(state.id, signal.type)
        if cached is not None:
            return cached

        outcome = self.lookup_outcome(state.id, signal.type)
        self._cache_set(state.id, signal.type, outcome)
        return outcome
