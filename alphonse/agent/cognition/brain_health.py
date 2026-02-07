from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alphonse.agent.cognition.intent_catalog import IntentCatalogStore
from alphonse.agent.cognition.prompt_store import SqlitePromptStore


class BrainUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class BrainHealth:
    db_path: str
    catalog_available: bool
    prompt_store_available: bool


def check_brain_health(db_path: str | Path) -> BrainHealth:
    resolved = str(db_path)
    catalog_available = IntentCatalogStore(resolved).is_available()
    prompt_store_available = SqlitePromptStore(resolved).is_available()
    return BrainHealth(
        db_path=resolved,
        catalog_available=catalog_available,
        prompt_store_available=prompt_store_available,
    )


def require_brain_health(db_path: str | Path) -> BrainHealth:
    health = check_brain_health(db_path)
    if not health.catalog_available:
        raise BrainUnavailable(
            f"intent catalog unavailable db_path={health.db_path}"
        )
    if not health.prompt_store_available:
        raise BrainUnavailable(
            f"prompt store unavailable db_path={health.db_path}"
        )
    return health

