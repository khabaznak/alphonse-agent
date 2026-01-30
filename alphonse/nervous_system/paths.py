from __future__ import annotations

import os
from pathlib import Path


def resolve_nervous_system_db_path() -> Path:
    default_path = Path(__file__).resolve().parent / "db" / "nerve-db"
    configured = os.getenv("NERVE_DB_PATH")
    if not configured:
        default_path.parent.mkdir(parents=True, exist_ok=True)
        return default_path
    configured_path = Path(configured)
    if configured_path.is_absolute():
        configured_path.parent.mkdir(parents=True, exist_ok=True)
        return configured_path
    resolved = (Path(__file__).resolve().parent.parent / configured_path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
