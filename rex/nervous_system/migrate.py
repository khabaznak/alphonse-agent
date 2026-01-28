"""Apply schema to the nervous system SQLite database."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


def default_db_path() -> Path:
    return Path(__file__).resolve().parent / "db" / "nerve-db"


def apply_schema(db_path: Path) -> None:
    schema_path = Path(__file__).resolve().parent / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_sql)


def main() -> None:
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    apply_schema(db_path)
    print(f"Applied schema to {db_path}")


if __name__ == "__main__":
    main()
