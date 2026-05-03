from __future__ import annotations

import sqlite3
from pathlib import Path

from alphonse.agent.actions.runtime import build_default_action_registry
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.seed import apply_seed


def test_action_registry_is_deterministic_only() -> None:
    keys = set(build_default_action_registry().list_keys())
    assert keys == {
        "handle_conscious_message",
        "handle_pdca_dispatch_kick",
        "handle_pdca_failure_notice",
        "handle_pdca_slice_request",
        "handle_timed_dispatch",
        "shutdown",
    }


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

        runtime_escalation_route = conn.execute(
            """
            SELECT COUNT(*)
            FROM transitions t
            JOIN signals s ON s.id = t.signal_id
            WHERE t.is_enabled = 1
              AND s.key = 'sense.runtime.message.user.received'
              AND t.action_key = 'handle_conscious_message'
            """
        ).fetchone()
        assert int(runtime_escalation_route[0] or 0) > 0

        legacy = conn.execute(
            "SELECT COUNT(*) FROM transitions WHERE action_key = 'handle_incoming_message' AND is_enabled = 1"
        ).fetchone()
        assert int(legacy[0] or 0) == 0

        removed_actions = conn.execute(
            """
            SELECT COUNT(*)
            FROM transitions
            WHERE is_enabled = 1
              AND action_key IN (
                'handle_telegram_invite',
                'system_reminder',
                'handle_action_failure',
                'handle_timed_signals_query',
                'handle_subconscious_prompt'
              )
            """
        ).fetchone()
        assert int(removed_actions[0] or 0) == 0

        notice_routes = conn.execute(
            """
            SELECT COUNT(*)
            FROM transitions t
            JOIN signals s ON s.id = t.signal_id
            WHERE t.is_enabled = 1
              AND s.key = 'pdca.failed'
              AND t.action_key = 'handle_pdca_failure_notice'
            """
        ).fetchone()
        assert int(notice_routes[0] or 0) > 0

        generic_action_outcomes = conn.execute(
            """
            SELECT COUNT(*)
            FROM signals
            WHERE key IN ('action.succeeded', 'action.failed')
            """
        ).fetchone()
        assert int(generic_action_outcomes[0] or 0) == 0
