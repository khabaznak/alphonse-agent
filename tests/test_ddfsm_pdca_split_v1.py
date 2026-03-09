from __future__ import annotations

import sqlite3
from pathlib import Path

from alphonse.agent.cognition.intentions.intent_pipeline import build_default_pipeline_with_bus
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.seed import apply_seed
from alphonse.agent.nervous_system.senses.bus import Bus


def test_intent_pipeline_registry_is_deterministic_only() -> None:
    pipeline = build_default_pipeline_with_bus(Bus())
    keys = set(pipeline.actions.list_keys())
    assert "handle_conscious_message" in keys
    assert "handle_timed_dispatch" in keys
    assert "handle_subconscious_prompt" in keys
    assert "shutdown" in keys
    assert "handle_incoming_message" not in keys
    assert "handle_status" not in keys
    assert "handle_pdca_slice_request" not in keys


def test_seed_routes_conscious_and_status_removed(tmp_path: Path) -> None:
    db_path = tmp_path / "nervous_system.db"
    apply_schema(db_path)
    apply_seed(db_path)
    with sqlite3.connect(db_path) as conn:
        status_enabled = conn.execute(
            "SELECT is_enabled FROM signals WHERE key = 'api.status_requested'"
        ).fetchone()
        assert status_enabled is None

        terminal_routes = conn.execute(
            """
            SELECT COUNT(*)
            FROM transitions t
            JOIN signals s ON s.id = t.signal_id
            WHERE t.is_enabled = 1
              AND s.key IN ('terminal.command_updated', 'terminal.command_executed')
              AND t.action_key = 'handle_conscious_message'
            """
        ).fetchone()
        assert int(terminal_routes[0] or 0) > 0

        legacy = conn.execute(
            "SELECT COUNT(*) FROM transitions WHERE action_key = 'handle_incoming_message' AND is_enabled = 1"
        ).fetchone()
        assert int(legacy[0] or 0) == 0
