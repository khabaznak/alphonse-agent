"""Data-driven finite state machine (FSM) backed by SQLite."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rex.senses.bus import Signal as BusSignal


@dataclass
class DDFSMConfig:
    db_path: str
    queries_path: str | None = None


@dataclass
class State:
    id: int
    name: str
    key: str
    policy_key: str | None = None


@dataclass
class CurrentState:
    id: int
    key: str | None = None
    name: str | None = None


class DDFSM:
    def __init__(self, config: DDFSMConfig) -> None:
        self.config = config
        self._queries = self._load_queries()

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

    def execute(self, query_key: str, params: dict[str, Any]) -> list[sqlite3.Row]:
        sql = self._queries.get(query_key)
        if not sql:
            raise KeyError(f"Query not found: {query_key}")
        with self._connect() as conn:
            cur = conn.execute(sql, params)
            return cur.fetchall()

    def get_transition(self, state_id: int, signal_id: int) -> sqlite3.Row | None:
        rows = self.execute(
            "transition_lookup",
            {"state_id": state_id, "signal_id": signal_id},
        )
        return rows[0] if rows else None

    def next_state(self, state_id: int, signal_key: str) -> State | None:
        rows = self.execute(
            "transition_next_state_by_signal_key",
            {"state_id": state_id, "signal_key": signal_key},
        )
        if not rows:
            return None
        row = rows[0]
        return State(
            id=row["state_id"],
            name=row["state_name"],
            key=row["state_key"],
            policy_key=row["policy_key"],
        )

    def handle(self, state: CurrentState, signal: BusSignal, ctx: object) -> State | None:
        # ctx is reserved for future policy/guard evaluation.
        _ = ctx
        return self.next_state(state.id, signal.type)
