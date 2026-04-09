from __future__ import annotations

import sqlite3
from pathlib import Path

from alphonse.agent import cli
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.pdca_queue_store import append_pdca_event
from alphonse.agent.nervous_system.pdca_queue_store import describe_pdca_runtime_flush_counts
from alphonse.agent.nervous_system.pdca_queue_store import save_pdca_checkpoint
from alphonse.agent.nervous_system.pdca_queue_store import upsert_pdca_task


def _seed_runtime_rows(db_path: Path) -> None:
    apply_schema(db_path)
    task_id = upsert_pdca_task(
        {
            "owner_id": "u1",
            "conversation_key": "telegram:seed",
            "status": "running",
        }
    )
    _ = save_pdca_checkpoint(
        task_id=task_id,
        state={"a": 1, "cycle_index": 1},
        expected_version=0,
    )
    _ = append_pdca_event(task_id=task_id, event_type="slice.requested", payload={"seeded": True})
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO signal_queue (signal_id, signal_type, payload, source, durable)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("sig-seed", "pdca.slice.requested", "{}", "test", 1),
        )
        conn.commit()


def test_agent_flush_removes_pdca_and_signal_queue_rows(tmp_path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    _seed_runtime_rows(db_path)
    monkeypatch.setattr(cli, "_runtime_appears_active", lambda: False)

    parser = cli.build_parser()
    args = parser.parse_args(["agent", "flush"])
    cli._dispatch_command(args, db_path, parser)

    out = capsys.readouterr().out
    assert "Agent flush completed" in out
    counts = describe_pdca_runtime_flush_counts(include_signal_queue=True)
    assert counts == {
        "pdca_events": 0,
        "pdca_checkpoints": 0,
        "pdca_tasks": 0,
        "signal_queue": 0,
    }


def test_agent_flush_blocks_when_runtime_appears_active_without_force(tmp_path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    _seed_runtime_rows(db_path)
    monkeypatch.setattr(cli, "_runtime_appears_active", lambda: True)

    parser = cli.build_parser()
    args = parser.parse_args(["agent", "flush"])
    cli._dispatch_command(args, db_path, parser)

    out = capsys.readouterr().out
    assert "Refusing to flush while agent runtime appears active" in out
    counts = describe_pdca_runtime_flush_counts(include_signal_queue=True)
    assert counts["pdca_tasks"] == 1
    assert counts["signal_queue"] == 1


def test_agent_flush_dry_run_does_not_delete_rows(tmp_path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    _seed_runtime_rows(db_path)
    monkeypatch.setattr(cli, "_runtime_appears_active", lambda: False)

    parser = cli.build_parser()
    args = parser.parse_args(["agent", "flush", "--dry-run"])
    cli._dispatch_command(args, db_path, parser)

    out = capsys.readouterr().out
    assert "Agent flush dry run" in out
    counts = describe_pdca_runtime_flush_counts(include_signal_queue=True)
    assert counts["pdca_events"] == 1
    assert counts["pdca_checkpoints"] == 1
    assert counts["pdca_tasks"] == 1
    assert counts["signal_queue"] == 1


def test_agent_flush_force_overrides_runtime_activity_check(tmp_path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    _seed_runtime_rows(db_path)
    monkeypatch.setattr(cli, "_runtime_appears_active", lambda: True)

    parser = cli.build_parser()
    args = parser.parse_args(["agent", "flush", "--force"])
    cli._dispatch_command(args, db_path, parser)

    out = capsys.readouterr().out
    assert "Agent flush completed" in out
    counts = describe_pdca_runtime_flush_counts(include_signal_queue=True)
    assert counts == {
        "pdca_events": 0,
        "pdca_checkpoints": 0,
        "pdca_tasks": 0,
        "signal_queue": 0,
    }
