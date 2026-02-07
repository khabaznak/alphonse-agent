from __future__ import annotations

import sqlite3
from pathlib import Path


def apply_seed(db_path: Path) -> None:
    seed_path = Path(__file__).resolve().parent / "seed.sql"
    if not seed_path.exists():
        return
    seed_sql = seed_path.read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(seed_sql)
