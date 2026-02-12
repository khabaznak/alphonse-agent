from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alphonse.agent.cognition.prompt_templates_runtime import PROMPT_SEEDS_DIR


class BrainUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class BrainHealth:
    db_path: str
    prompt_store_available: bool


def check_brain_health(db_path: str | Path) -> BrainHealth:
    resolved = str(db_path)
    prompt_store_available = bool(PROMPT_SEEDS_DIR.exists() and any(PROMPT_SEEDS_DIR.iterdir()))
    return BrainHealth(
        db_path=resolved,
        prompt_store_available=prompt_store_available,
    )


def require_brain_health(db_path: str | Path) -> BrainHealth:
    health = check_brain_health(db_path)
    if not health.prompt_store_available:
        raise BrainUnavailable(
            f"prompt templates unavailable seeds_dir={PROMPT_SEEDS_DIR} db_path={health.db_path}"
        )
    return health
