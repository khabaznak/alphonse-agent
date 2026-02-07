from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


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
    alphonse_root = Path(__file__).resolve().parents[2]
    relative_parts = [part for part in configured_path.parts if part not in ("..", ".")]
    normalized_relative = Path(*relative_parts) if relative_parts else configured_path
    resolved = (alphonse_root / normalized_relative).resolve()
    logger.info(
        "Resolved relative NERVE_DB_PATH to %s (root=%s)",
        resolved,
        alphonse_root,
    )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
