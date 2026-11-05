import json
from pathlib import Path
import sqlite3

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.memory.daily_ledger import DailyLedgerProjector
from alphonse.agent_v2.core.memory.migration import RECEIPT, backup, migrate_scope, restore


@pytest.fixture
def scope(tmp_path):
    root = tmp_path / "memory"
    root.mkdir()
    (root / "ledger-0001.md").write_text("Original historical audit; not proof of current state.")
    (root / "project-summary.md").write_text("An unsupported derived fact.")
    session = root / "sessions" / "old-session"
    session.mkdir(parents=True)
    (session / "summary.md").write_text("Old summary with no resolvable canonical source.")
    return root


def inventory(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_dry_run_is_read_only_and_apply_is_additive_idempotent(scope):
    before = inventory(scope)
    dry = migrate_scope(scope, owner="alex", project="old")
    assert inventory(scope) == before
    assert dry["dry_run"] and dry["added"] == 3
    applied = migrate_scope(scope, owner="alex", project="old", apply=True)
    assert applied["files"] == dry["files"]
    assert applied["copied_audits"] == 1
    after = inventory(scope)
    assert all(after[name] == content for name, content in before.items())
    assert after["legacy/ledger-0001.md"] == before["ledger-0001.md"]
    repeated = migrate_scope(scope, owner="alex", project="old", apply=True)
    assert repeated["added"] == repeated["copied_audits"] == 0
    assert inventory(scope) == after
    candidates = repeated["files"]
    assert len({c["candidate_id"] for c in candidates}) == 3
    assert all(c["basis"] == "legacy_unverified" and c["requires_reconciliation"]
               and c["visibility"] == "private" and c["owner_user_id"] == "alex" for c in candidates)
    assert {c["kind"] for c in candidates} == {"audit", "derived_summary"}
    assert all("observed_at" not in c and "speaker_id" not in c for c in candidates)


def test_partial_migration_adopts_identical_copy(scope):
    (scope / "legacy").mkdir()
    original = (scope / "ledger-0001.md").read_bytes()
    (scope / "legacy/ledger-0001.md").write_bytes(original)
    result = migrate_scope(scope, owner="alex", project="p", apply=True)
    assert result["added"] == 3
    assert result["copied_audits"] == 0
    assert len([entry for entry in result["files"] if entry["kind"] == "audit"]) == 1


def test_conflicts_fail_before_any_copy_and_preserve_both(scope):
    (scope / "legacy").mkdir()
    (scope / "legacy/ledger-0001.md").write_text("Conflicting audit")
    before = inventory(scope)
    with pytest.raises(ValueError, match="archive_conflict"):
        migrate_scope(scope, owner="alex", project="p", apply=True)
    assert inventory(scope) == before


@pytest.mark.parametrize("change", ["source", "archive", "missing", "owner", "future_version"])
def test_receipt_never_guesses_changed_or_ambiguous_state(scope, change):
    migrate_scope(scope, owner="alex", project="p", apply=True)
    if change in {"source", "archive"}:
        target = "ledger-0001.md" if change == "source" else "legacy/ledger-0001.md"
        (scope / target).write_text("Changed")
    elif change == "missing":
        (scope / "ledger-0001.md").unlink()
    elif change == "future_version":
        receipt = json.loads((scope / RECEIPT).read_text())
        receipt["version"] = 999
        (scope / RECEIPT).write_text(json.dumps(receipt))
    before = inventory(scope)
    with pytest.raises(ValueError, match="reconciliation_required|receipt_mismatch"):
        migrate_scope(scope, owner="other" if change == "owner" else "alex", project="p", apply=True)
    assert inventory(scope) == before


def test_migration_rejects_symlink_sources_and_destinations(scope, tmp_path):
    secret = tmp_path / "private"
    secret.mkdir()
    (secret / "ledger-0002.md").write_text("Private sentinel")
    linked = scope / "ledger-0002.md"
    linked.symlink_to(secret / linked.name)
    with pytest.raises(ValueError, match="symlink"):
        migrate_scope(scope, owner="alex", project="p", apply=True)
    linked.unlink()
    (scope / "legacy").symlink_to(secret, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        migrate_scope(scope, owner="alex", project="p", apply=True)
    assert not (scope / RECEIPT).exists()


def test_backup_restore_includes_wal_and_never_overwrites_new_data(scope, tmp_path):
    database = tmp_path / "state.sqlite3"
    writer = sqlite3.connect(database)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE events (id TEXT PRIMARY KEY, content TEXT)")
    writer.execute("INSERT INTO events VALUES ('old', 'Original')")
    writer.commit()
    checkpoint = tmp_path / "checkpoint"
    manifest = backup(database, [scope], checkpoint)
    assert "database.sqlite3" in manifest["files"]
    assert "roots/0/project-summary.md" in manifest["files"]
    migrate_scope(scope, owner="alex", project="p", apply=True)
    writer.execute("INSERT INTO events VALUES ('new', 'Compatible new event')")
    writer.commit()
    recovered = tmp_path / "recovered"
    assert restore(checkpoint, recovered) == manifest
    with sqlite3.connect(recovered / "database.sqlite3") as conn:
        assert conn.execute("SELECT id FROM events").fetchall() == [("old",)]
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert not (recovered / "roots/0" / RECEIPT).exists()
    assert (scope / RECEIPT).exists()
    assert writer.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2
    with pytest.raises(ValueError, match="destination_exists"):
        restore(checkpoint, scope)
    writer.close()


def test_corrupt_backup_and_unsafe_destinations_fail_closed(scope, tmp_path):
    database = tmp_path / "db.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE test (value)")
    with pytest.raises(ValueError, match="inside_source"):
        backup(database, [scope], scope / "backup")
    checkpoint = tmp_path / "backup"
    backup(database, [scope], checkpoint)
    (checkpoint / "roots/0/project-summary.md").write_text("Tampered")
    with pytest.raises(ValueError, match="integrity_failed"):
        restore(checkpoint, tmp_path / "recovery")
    assert not (tmp_path / "recovery").exists()


def test_restore_does_not_accept_checkpoint_changed_during_copy(scope, tmp_path, monkeypatch):
    from alphonse.agent_v2.core.memory import migration
    database = tmp_path / "db.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE test (value)")
    checkpoint = tmp_path / "backup"
    backup(database, [scope], checkpoint)
    original = migration.shutil.copytree

    def changed_copy(source, destination, *args, **kwargs):
        if Path(source) == checkpoint:
            (checkpoint / "roots/0/project-summary.md").write_text("Changed after validation")
        return original(source, destination, *args, **kwargs)

    monkeypatch.setattr(migration.shutil, "copytree", changed_copy)
    with pytest.raises(ValueError, match="changed_during_restore"):
        restore(checkpoint, tmp_path / "recovery")
    assert not (tmp_path / "recovery").exists()


def test_repeated_migration_preserves_canonical_events_knowledge_and_daily_views(scope, tmp_path):
    from alphonse.agent_v2.core.memory.knowledge import KnowledgeStore
    database = tmp_path / "db.sqlite"
    conversations = SQLiteConversationStore(database)
    KnowledgeStore(str(database))
    event = conversations.record(owner_user_id="alex", content="Recordatorio para mañana",
                                 role="user", source="synthetic", source_message_id="unique",
                                 created_at="2026-10-18T05:30:00+00:00")
    projector = DailyLedgerProjector(conversation_store=conversations, users_root=lambda: tmp_path / "users",
                                     timezone_provider=lambda _: "America/Mexico_City")
    projector.refresh_for_event(event)
    daily_before = inventory(tmp_path / "users")
    events_before = [e.to_dict() for e in conversations.list_all(owner_user_id="alex")]
    for _ in range(3):
        migrate_scope(scope, owner="alex", project="p", apply=True)
        projector.refresh_for_event(event)
    assert [e.to_dict() for e in conversations.list_all(owner_user_id="alex")] == events_before
    assert inventory(tmp_path / "users") == daily_before
    assert "2026-10-17.md" in str(projector.refresh_for_event(event))
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM memory_knowledge").fetchone()[0] == 0
