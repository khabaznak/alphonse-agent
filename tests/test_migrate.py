from __future__ import annotations

import sqlite3
from pathlib import Path

from alphonse.agent.nervous_system.migrate import apply_schema


def test_apply_schema_drops_legacy_capability_gap_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "nerve-db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE capability_gaps (gap_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE gap_proposals (id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE gap_tasks (id TEXT PRIMARY KEY)")

    apply_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert "capability_gaps" not in names
    assert "gap_proposals" not in names
    assert "gap_tasks" not in names
