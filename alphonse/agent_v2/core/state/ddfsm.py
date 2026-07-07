"""V2-native data-driven finite state machine."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from threading import RLock

AVAILABLE = "available"
WORKING = "working"
WAITING = "waiting"
ERROR = "error"

MESSAGE_DEQUEUED = "message_dequeued"
PROCESSOR_COMPLETED = "processor_completed"
PROCESSOR_WAITING = "processor_waiting"
PROCESSOR_RESUMED = "processor_resumed"
PROCESSOR_FAILED = "processor_failed"
ERROR_CLEARED = "error_cleared"
STOP_REQUESTED = "stop_requested"


@dataclass(frozen=True)
class DDFSMConfig:
    db_path: str = ":memory:"
    seed: bool = True


@dataclass(frozen=True)
class CurrentState:
    id: int
    key: str
    name: str


@dataclass(frozen=True)
class CoreSignal:
    key: str
    source: str = "core"


@dataclass(frozen=True)
class TransitionOutcome:
    matched: bool
    reason: str
    transition_id: int | None = None
    next_state_id: int | None = None
    next_state_key: str | None = None
    next_state_name: str | None = None


class DDFSM:
    """SQLite-backed transition lookup for the v2 core loop."""

    def __init__(
        self,
        config: DDFSMConfig | None = None,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self.config = config or DDFSMConfig()
        self._lock = RLock()
        self._connection = connection or sqlite3.connect(self.config.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        if self.config.seed:
            self._ensure_schema()
            self._seed_defaults()

    def current_state_for_key(self, key: str) -> CurrentState:
        with self._lock:
            row = self._connection.execute(
                "SELECT id, key, name FROM states WHERE key = ? AND is_enabled = 1",
                (key,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown_state:{key}")
        return CurrentState(id=int(row["id"]), key=str(row["key"]), name=str(row["name"]))

    def handle(self, state: CurrentState, signal: CoreSignal) -> TransitionOutcome:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT
                    t.id AS transition_id,
                    s.id AS next_state_id,
                    s.key AS next_state_key,
                    s.name AS next_state_name
                FROM transitions t
                JOIN states s ON s.id = t.next_state_id
                JOIN signals sig ON sig.id = t.signal_id
                WHERE t.is_enabled = 1
                  AND sig.is_enabled = 1
                  AND sig.key = ?
                  AND ((t.match_any_state = 0 AND t.state_id = ?) OR t.match_any_state = 1)
                ORDER BY t.match_any_state ASC, t.priority ASC, t.id ASC
                LIMIT 1
                """,
                (signal.key, state.id),
            ).fetchone()
        if row is None:
            return TransitionOutcome(matched=False, reason="NO_TRANSITION")
        return TransitionOutcome(
            matched=True,
            reason="MATCH",
            transition_id=int(row["transition_id"]),
            next_state_id=int(row["next_state_id"]),
            next_state_key=str(row["next_state_key"]),
            next_state_name=str(row["next_state_name"]),
        )

    def _ensure_schema(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS states (
                  id INTEGER PRIMARY KEY,
                  key TEXT NOT NULL UNIQUE,
                  name TEXT NOT NULL,
                  is_enabled INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS signals (
                  id INTEGER PRIMARY KEY,
                  key TEXT NOT NULL UNIQUE,
                  name TEXT NOT NULL,
                  is_enabled INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS transitions (
                  id INTEGER PRIMARY KEY,
                  state_id INTEGER NOT NULL,
                  signal_id INTEGER NOT NULL,
                  next_state_id INTEGER NOT NULL,
                  priority INTEGER NOT NULL DEFAULT 100,
                  is_enabled INTEGER NOT NULL DEFAULT 1,
                  match_any_state INTEGER NOT NULL DEFAULT 0,
                  FOREIGN KEY (state_id) REFERENCES states(id),
                  FOREIGN KEY (signal_id) REFERENCES signals(id),
                  FOREIGN KEY (next_state_id) REFERENCES states(id)
                );
                """
            )

    def _seed_defaults(self) -> None:
        states = (
            (1, AVAILABLE, "Available"),
            (2, WORKING, "Working"),
            (3, WAITING, "Waiting"),
            (4, ERROR, "Error"),
        )
        signals = (
            (1, MESSAGE_DEQUEUED, "Message Dequeued"),
            (2, PROCESSOR_COMPLETED, "Processor Completed"),
            (3, PROCESSOR_WAITING, "Processor Waiting"),
            (4, PROCESSOR_RESUMED, "Processor Resumed"),
            (5, PROCESSOR_FAILED, "Processor Failed"),
            (6, ERROR_CLEARED, "Error Cleared"),
            (7, STOP_REQUESTED, "Stop Requested"),
        )
        transitions = (
            (1, AVAILABLE, MESSAGE_DEQUEUED, WORKING, 10, 0),
            (2, WORKING, PROCESSOR_COMPLETED, AVAILABLE, 10, 0),
            (3, WORKING, PROCESSOR_WAITING, WAITING, 10, 0),
            (4, WAITING, PROCESSOR_RESUMED, WORKING, 10, 0),
            (5, AVAILABLE, PROCESSOR_FAILED, ERROR, 10, 1),
            (6, ERROR, ERROR_CLEARED, AVAILABLE, 10, 0),
        )
        with self._lock:
            with self._connection:
                self._connection.executemany(
                    "INSERT OR IGNORE INTO states (id, key, name) VALUES (?, ?, ?)",
                    states,
                )
                self._connection.executemany(
                    "INSERT OR IGNORE INTO signals (id, key, name) VALUES (?, ?, ?)",
                    signals,
                )
                for transition_id, from_key, signal_key, to_key, priority, any_state in transitions:
                    self._connection.execute(
                        """
                        INSERT OR IGNORE INTO transitions (
                          id, state_id, signal_id, next_state_id, priority, match_any_state
                        )
                        SELECT ?, from_state.id, sig.id, to_state.id, ?, ?
                        FROM states from_state
                        JOIN signals sig ON sig.key = ?
                        JOIN states to_state ON to_state.key = ?
                        WHERE from_state.key = ?
                        """,
                        (transition_id, priority, any_state, signal_key, to_key, from_key),
                    )


def build_default_ddfsm() -> DDFSM:
    return DDFSM()
