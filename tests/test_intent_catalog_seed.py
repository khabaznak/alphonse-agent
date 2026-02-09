from __future__ import annotations

import sqlite3
from pathlib import Path

from alphonse.agent.cognition.intent_catalog import IntentCatalogStore, seed_default_intents
from alphonse.agent.nervous_system.migrate import apply_schema


def _setup_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "nerve-db"
    apply_schema(db_path)
    return db_path


def test_seed_inserts_factory_intents(tmp_path: Path) -> None:
    db_path = _setup_db(tmp_path)
    store = IntentCatalogStore(str(db_path))
    intents = {spec.intent_name: spec for spec in store.list_all()}
    assert "greeting" in intents
    assert intents["greeting"].origin == "factory"
    assert intents["greeting"].intent_version == "1.0.0"
    assert "core.identity.query_agent_name" in intents
    assert "core.identity.query_user_name" in intents
    assert "core.onboarding.start" in intents


def test_seed_preserves_enabled_flag(tmp_path: Path) -> None:
    db_path = _setup_db(tmp_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE intent_specs SET enabled = 0 WHERE intent_name = 'help'"
        )
    seed_default_intents(str(db_path))
    store = IntentCatalogStore(str(db_path))
    spec = store.get("help")
    assert spec is not None
    assert spec.enabled is False
