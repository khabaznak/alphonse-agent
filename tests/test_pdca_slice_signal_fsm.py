from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from alphonse.agent.nervous_system.ddfsm import DDFSM, DDFSMConfig
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.seed import apply_seed


def test_fsm_has_transition_for_pdca_slice_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    apply_seed(db_path)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT id FROM states WHERE key = 'idle'").fetchone()
    assert row is not None
    idle_state_id = int(row[0])

    fsm = DDFSM(DDFSMConfig(db_path=str(db_path)))
    outcome = fsm.lookup_outcome(state_id=idle_state_id, signal_key="pdca.slice.requested")
    assert outcome.matched is True
    assert outcome.action_key == "handle_pdca_slice_request"
    assert outcome.next_state_key == "rehydrating_slice"


def test_fsm_pdca_lifecycle_transitions_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    apply_seed(db_path)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT key, id FROM states").fetchall()
        dnd = conn.execute("SELECT is_enabled FROM states WHERE key = 'dnd'").fetchone()
    state_ids = {str(row[0]): int(row[1]) for row in rows}
    assert dnd is not None
    assert int(dnd[0]) == 0

    fsm = DDFSM(DDFSMConfig(db_path=str(db_path)))

    assert fsm.lookup_outcome(state_ids["rehydrating_slice"], "pdca.slice.requested").action_key == "handle_pdca_slice_request"
    assert fsm.lookup_outcome(state_ids["rehydrating_slice"], "pdca.slice.persisted").next_state_key == "idle"
    assert fsm.lookup_outcome(state_ids["rehydrating_slice"], "pdca.waiting_user").next_state_key == "waiting_user"
    assert fsm.lookup_outcome(state_ids["rehydrating_slice"], "pdca.slice.completed").next_state_key == "idle"
    assert fsm.lookup_outcome(state_ids["waiting_user"], "sense.api.message.user.received").next_state_key == "rehydrating_slice"
