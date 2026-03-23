from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system import operational_facts as facts_store


def _setup_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    return db_path


def test_operational_facts_table_and_indexes_exist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = _setup_db(monkeypatch, tmp_path)
    with sqlite3.connect(db_path) as conn:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(operational_facts)").fetchall()}
        assert "id" in columns
        assert "key" in columns
        assert "fact_type" in columns
        assert "scope" in columns
        assert "created_by" in columns
        assert "confidence" in columns
        indexes = {str(row[1]) for row in conn.execute("PRAGMA index_list(operational_facts)").fetchall()}
        assert "idx_operational_facts_key_unique" in indexes
        assert "idx_operational_facts_created_by_scope" in indexes
        assert "idx_operational_facts_type_status_updated" in indexes
        assert "idx_operational_facts_updated_at" in indexes


def test_upsert_updates_same_key_and_json_round_trips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _setup_db(monkeypatch, tmp_path)
    created = facts_store.upsert_operational_fact(
        created_by="user_a",
        key="ops.router.main",
        title="Main router",
        fact_type="system_asset",
        content_json={"ip": "192.168.1.1", "vendor": "Ubiquiti"},
        tags=["network", "edge"],
        scope="private",
        confidence=0.8,
    )
    assert created["key"] == "ops.router.main"
    assert created["content_json"]["ip"] == "192.168.1.1"
    assert "network" in created["tags"]

    updated = facts_store.upsert_operational_fact(
        created_by="user_a",
        key="ops.router.main",
        title="Main router updated",
        fact_type="system_asset",
        content_json={"ip": "192.168.1.254"},
        tags=["network", "core"],
        scope="global",
        confidence=1.0,
    )
    assert updated["id"] == created["id"]
    assert updated["title"] == "Main router updated"
    assert updated["scope"] == "global"
    assert updated["content_json"]["ip"] == "192.168.1.254"
    assert "core" in updated["tags"]


def test_search_visibility_rules_private_and_global(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _setup_db(monkeypatch, tmp_path)
    facts_store.upsert_operational_fact(
        created_by="user_a",
        key="ops.private.a",
        title="Private A",
        fact_type="workflow_rule",
        scope="private",
    )
    facts_store.upsert_operational_fact(
        created_by="user_a",
        key="ops.global.a",
        title="Global A",
        fact_type="workflow_rule",
        scope="global",
    )
    facts_store.upsert_operational_fact(
        created_by="user_b",
        key="ops.private.b",
        title="Private B",
        fact_type="workflow_rule",
        scope="private",
    )

    as_user_a = facts_store.search_operational_facts(created_by="user_a")
    as_user_b = facts_store.search_operational_facts(created_by="user_b")
    keys_a = {row["key"] for row in as_user_a}
    keys_b = {row["key"] for row in as_user_b}
    assert "ops.private.a" in keys_a
    assert "ops.private.b" not in keys_a
    assert "ops.global.a" in keys_a
    assert "ops.private.a" not in keys_b
    assert "ops.private.b" in keys_b
    assert "ops.global.a" in keys_b


def test_remove_hard_delete_with_owner_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _setup_db(monkeypatch, tmp_path)
    created = facts_store.upsert_operational_fact(
        created_by="user_a",
        key="ops.to.delete",
        title="Delete me",
        fact_type="integration_note",
    )
    denied = facts_store.remove_operational_fact(created_by="user_b", key="ops.to.delete")
    assert denied is False
    still_there = facts_store.search_operational_facts(created_by="user_a", query="delete")
    assert any(row["id"] == created["id"] for row in still_there)

    allowed = facts_store.remove_operational_fact(created_by="user_a", key="ops.to.delete")
    assert allowed is True
    gone = facts_store.search_operational_facts(created_by="user_a", query="delete")
    assert all(row["id"] != created["id"] for row in gone)


def test_validation_rejects_invalid_fact_type_scope_and_confidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = _setup_db(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="invalid_fact_type"):
        facts_store.upsert_operational_fact(
            created_by="user_a",
            key="ops.bad.type",
            title="Bad",
            fact_type="bad_type",
        )
    with pytest.raises(ValueError, match="invalid_scope"):
        facts_store.upsert_operational_fact(
            created_by="user_a",
            key="ops.bad.scope",
            title="Bad",
            fact_type="system_asset",
            scope="team",
        )
    with pytest.raises(ValueError, match="invalid_confidence"):
        facts_store.upsert_operational_fact(
            created_by="user_a",
            key="ops.bad.confidence",
            title="Bad",
            fact_type="system_asset",
            confidence=1.5,
        )

    with sqlite3.connect(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO operational_facts (
                    id, key, title, fact_type, scope, created_by
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("id-1", "ops.db.bad", "Bad", "not_allowed", "private", "user_a"),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO operational_facts (
                    id, key, title, fact_type, scope, created_by, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("id-2", "ops.db.bad.conf", "Bad", "system_asset", "private", "user_a", 9.2),
            )
