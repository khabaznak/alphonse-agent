from __future__ import annotations

from alphonse.agent.observability.log_manager import get_component_logger
import os
from pathlib import Path

logger = get_component_logger("nervous_system.paths")


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


def resolve_observability_db_path() -> Path:
    default_path = Path(__file__).resolve().parent / "db" / "observability.db"
    configured = os.getenv("ALPHONSE_OBSERVABILITY_DB_PATH")
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
        "Resolved relative ALPHONSE_OBSERVABILITY_DB_PATH to %s (root=%s)",
        resolved,
        alphonse_root,
    )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
