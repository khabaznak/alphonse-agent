from __future__ import annotations

import sqlite3
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.task_state import EMPTY_MARKDOWN, TaskState
from alphonse.agent_v2.core.io.channels import ChannelAddress
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.database import DEFAULT_BUSY_TIMEOUT_MS, connect_database, default_database_path
from alphonse.agent_v2.retention import prune_operational_data
from alphonse.agent_v2.storage_migration import _copy_table, migrate_legacy_databases
from alphonse.agent_v2.users import V2UserStore


def test_shared_database_configuration_and_defaults(monkeypatch, tmp_path) -> None:
    database = tmp_path / "alphonse-v2.sqlite3"
    monkeypatch.setenv("ALPHONSE_V2_DB_PATH", str(database))
    monkeypatch.setenv("ALPHONSE_V2_USERS_DB_PATH", str(tmp_path / "legacy-users.sqlite3"))

    stores = (
        V2UserStore.default(),
        SQLiteQuestionStore.default(),
        SQLiteOutboundStore.default(),
        SQLiteConversationStore.default(),
    )

    assert default_database_path() == database
    assert {store.db_path for store in stores} == {str(database)}
    with connect_database(database) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == DEFAULT_BUSY_TIMEOUT_MS
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_schema_migrations'"
        ).fetchone()


def test_legacy_migration_is_backed_up_validated_and_idempotent(tmp_path) -> None:
    source = tmp_path / "v2-users.sqlite3"
    question_source = tmp_path / "v2-questions.sqlite3"
    outbox_source = tmp_path / "v2-outbox.sqlite3"
    target = tmp_path / "alphonse-v2.sqlite3"
    users = V2UserStore(source)
    users.set_users_root(tmp_path / "users")
    users.create_user(display_name="Alex", role="admin", user_id="alex")
    questions = SQLiteQuestionStore(question_source)
    question = questions.create_question(
        task=TaskState(task_id="parent-task", user="alex", project_id="alpha", goal="Ask Gaby"),
        question="Which option?",
        respondent_user_id="gaby",
    )
    questions.bind_delivery_metadata(
        question_id=question.question_id,
        metadata={"provider_message_id": "question-provider-id"},
    )
    outbox = SQLiteOutboundStore(outbox_source)
    delivery = outbox.enqueue(
        address=ChannelAddress("telegram", "telegram", "chat-1", alphonse_user_id="alex"),
        message="Delivery",
        correlation_id="correlation-one",
        project_id="alpha",
    )
    outbox.mark_delivered(delivery.outbox_message_id, provider_message_id="delivery-provider-id")
    sources = [source, question_source, outbox_source]

    first = migrate_legacy_databases(
        target=target,
        sources=sources,
        backup_root=tmp_path / "backups",
    )
    second = migrate_legacy_databases(
        target=target,
        sources=sources,
        backup_root=tmp_path / "backups",
    )

    assert first["status"] == "migrated"
    assert second["status"] == "already_applied"
    assert V2UserStore(target).get_user("alex") is not None
    migrated_question = SQLiteQuestionStore(target).get_question(question.question_id)
    assert migrated_question is not None
    assert migrated_question.status == "pending"
    assert migrated_question.metadata["delivery"]["provider_message_id"] == "question-provider-id"
    assert SQLiteQuestionStore(target).load_task_checkpoint("parent-task") is not None
    with connect_database(target) as connection:
        dependency = connection.execute(
            "SELECT * FROM v2_task_dependencies WHERE question_id=?",
            (question.question_id,),
        ).fetchone()
    assert dependency is not None
    migrated_delivery = SQLiteOutboundStore(target).get(delivery.outbox_message_id)
    assert migrated_delivery is not None
    assert migrated_delivery.status == "delivered"
    assert migrated_delivery.provider_message_id == "delivery-provider-id"
    backup_directories = list((tmp_path / "backups").glob("pre-unified-*"))
    assert len(backup_directories) == 1
    backup_directory = backup_directories[0]
    assert {item.name for item in sources}.issubset({item.name for item in backup_directory.iterdir()})
    with connect_database(target) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        migrations = {row[0] for row in connection.execute("SELECT migration_id FROM v2_schema_migrations")}
        assert migrations == {
            "2026-07-30-unified-v2-database", "2026-07-30-unified-schema-v1",
            "event_execution_leases_v1",
        }


def test_failed_migration_does_not_modify_legacy_database(tmp_path) -> None:
    source = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE v2_unknown (id TEXT PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO v2_unknown VALUES ('one','still here')")

    with pytest.raises(RuntimeError, match="unified_schema_missing_table"):
        migrate_legacy_databases(
            target=tmp_path / "unified.sqlite3",
            sources=[source],
            backup_root=tmp_path / "backups",
        )

    with sqlite3.connect(source) as connection:
        assert connection.execute("SELECT value FROM v2_unknown WHERE id='one'").fetchone()[0] == "still here"


def test_migration_refuses_colliding_backup_names_without_overwriting_sources(tmp_path) -> None:
    sources = [tmp_path / directory / 'legacy.sqlite3' for directory in ('first', 'second')]
    for index, source in enumerate(sources):
        source.parent.mkdir()
        with sqlite3.connect(source) as connection:
            connection.execute('CREATE TABLE fixture (value TEXT)')
            connection.execute('INSERT INTO fixture VALUES (?)', (str(index),))

    with pytest.raises(RuntimeError, match='unified_migration_backup_name_conflict'):
        migrate_legacy_databases(target=tmp_path / 'target.sqlite3', sources=sources,
                                 backup_root=tmp_path / 'backups')
    assert not (tmp_path / 'backups').exists()
    for index, source in enumerate(sources):
        with sqlite3.connect(source) as connection:
            assert connection.execute('SELECT value FROM fixture').fetchone()[0] == str(index)


@pytest.mark.parametrize('second_value, should_conflict', [('same', False), ('different', True)])
def test_migration_checks_existing_primary_key_contents(tmp_path, second_value, should_conflict) -> None:
    sources = [tmp_path / 'first.sqlite3', tmp_path / 'second.sqlite3']
    for source, value in zip(sources, ('same', second_value)):
        with sqlite3.connect(source) as connection:
            connection.execute('CREATE TABLE v2_user_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)')
            connection.execute('INSERT INTO v2_user_settings VALUES (?, ?, ?)', ('fixture', value, 'synthetic'))
    target = tmp_path / 'target.sqlite3'
    if should_conflict:
        with pytest.raises(RuntimeError, match='unified_migration_row_conflict: v2_user_settings'):
            migrate_legacy_databases(target=target, sources=sources, backup_root=tmp_path / 'backups')
    else:
        assert migrate_legacy_databases(target=target, sources=sources, backup_root=tmp_path / 'backups')['status'] == 'migrated'
    with sqlite3.connect(target) as connection:
        assert connection.execute("SELECT value FROM v2_user_settings WHERE key='fixture'").fetchone()[0] == 'same'
        marker = connection.execute("SELECT 1 FROM v2_schema_migrations WHERE migration_id='2026-07-30-unified-v2-database'").fetchone()
        assert bool(marker) is not should_conflict
    for source, value in zip(sources, ('same', second_value)):
        with sqlite3.connect(source) as connection:
            assert connection.execute('SELECT value FROM v2_user_settings').fetchone()[0] == value


def _migration_fixture(tmp_path, *, source_sql, destination_sql, source_rows, target_rows=()):
    source = tmp_path / 'source.sqlite3'
    with sqlite3.connect(source) as connection:
        connection.execute(source_sql)
        connection.executemany('INSERT INTO v2_fixture VALUES (' + ','.join('?' for _ in source_rows[0]) + ')', source_rows)
    connection = sqlite3.connect(tmp_path / 'target.sqlite3')
    connection.execute(destination_sql)
    if target_rows:
        connection.executemany('INSERT INTO v2_fixture VALUES (' + ','.join('?' for _ in target_rows[0]) + ')', target_rows)
    connection.commit()
    connection.execute('ATTACH DATABASE ? AS legacy_0', (str(source),))
    return connection


def test_migration_rejects_partial_composite_key(tmp_path):
    connection = _migration_fixture(tmp_path,
        source_sql='CREATE TABLE v2_fixture (part_a TEXT, value TEXT)',
        destination_sql='CREATE TABLE v2_fixture (part_a TEXT, part_b TEXT DEFAULT "default", value TEXT, PRIMARY KEY (part_a, part_b))',
        source_rows=[('one', 'value')])
    with pytest.raises(RuntimeError, match='unified_migration_source_key_missing'), connection:
        _copy_table(connection, alias='legacy_0', table='v2_fixture')
    assert connection.execute('SELECT COUNT(*) FROM v2_fixture').fetchone()[0] == 0
    connection.close()


def test_migration_preserves_older_schema_with_new_optional_destination_column(tmp_path):
    connection = _migration_fixture(tmp_path,
        source_sql='CREATE TABLE v2_fixture (id TEXT PRIMARY KEY, value TEXT)',
        destination_sql='CREATE TABLE v2_fixture (id TEXT PRIMARY KEY, value TEXT, extra TEXT DEFAULT "new")',
        source_rows=[('one', 'value')])
    with connection:
        _copy_table(connection, alias='legacy_0', table='v2_fixture')
    assert connection.execute('SELECT * FROM v2_fixture').fetchone() == ('one', 'value', 'new')
    connection.close()


def test_migration_refuses_to_drop_source_only_columns(tmp_path):
    connection = _migration_fixture(tmp_path,
        source_sql='CREATE TABLE v2_fixture (id TEXT PRIMARY KEY, value TEXT, old_context TEXT)',
        destination_sql='CREATE TABLE v2_fixture (id TEXT PRIMARY KEY, value TEXT)',
        source_rows=[('one', 'value', 'preserve')])
    with pytest.raises(RuntimeError, match='unified_migration_source_columns_unmapped'), connection:
        _copy_table(connection, alias='legacy_0', table='v2_fixture')
    assert connection.execute('SELECT old_context FROM legacy_0.v2_fixture').fetchone()[0] == 'preserve'
    connection.close()


def test_migration_refuses_existing_non_keyed_rows(tmp_path):
    connection = _migration_fixture(tmp_path,
        source_sql='CREATE TABLE v2_fixture (value TEXT, context TEXT)',
        destination_sql='CREATE TABLE v2_fixture (value TEXT UNIQUE, context TEXT)',
        source_rows=[('one', 'source')], target_rows=[('one', 'different')])
    with pytest.raises(RuntimeError, match='unified_migration_unkeyed_target_not_empty'), connection:
        _copy_table(connection, alias='legacy_0', table='v2_fixture')
    assert connection.execute('SELECT context FROM v2_fixture').fetchone()[0] == 'different'
    connection.close()


def test_migration_checks_non_keyed_duplicate_multiplicity(tmp_path):
    connection = _migration_fixture(tmp_path,
        source_sql='CREATE TABLE v2_fixture (value TEXT, context TEXT)',
        destination_sql='CREATE TABLE v2_fixture (value TEXT UNIQUE, context TEXT)',
        source_rows=[('one', 'same'), ('one', 'same')])
    with pytest.raises(RuntimeError, match='unified_migration_row_validation_failed'), connection:
        _copy_table(connection, alias='legacy_0', table='v2_fixture')
    connection.close()


def test_migration_refuses_non_keyed_retry_without_duplicating_rows(tmp_path):
    connection = _migration_fixture(tmp_path,
        source_sql='CREATE TABLE v2_fixture (value TEXT, context TEXT)',
        destination_sql='CREATE TABLE v2_fixture (value TEXT, context TEXT)',
        source_rows=[('one', 'same'), ('one', 'same')])
    with connection:
        _copy_table(connection, alias='legacy_0', table='v2_fixture')
    with pytest.raises(RuntimeError, match='unified_migration_unkeyed_target_not_empty'), connection:
        _copy_table(connection, alias='legacy_0', table='v2_fixture')
    assert connection.execute('SELECT value, context FROM v2_fixture').fetchall() == [('one', 'same'), ('one', 'same')]
    connection.close()


def test_populated_legacy_conversation_migrates_with_verified_index(tmp_path):
    source = tmp_path / 'conversations.sqlite3'
    store = SQLiteConversationStore(source)
    event = store.record(owner_user_id='owner', role='user', content='synthetic apple',
                         source='test', source_message_id='fixture')
    deleted = store.record(owner_user_id='owner', role='user', content='synthetic deleted',
                           source='test', source_message_id='deleted')
    assert store.delete(event_id=deleted.event_id, owner_user_id='owner')
    target = tmp_path / 'target.sqlite3'
    args = dict(target=target, sources=[source], backup_root=tmp_path / 'backups')
    assert migrate_legacy_databases(**args)['status'] == 'migrated'
    assert migrate_legacy_databases(**args)['status'] == 'already_applied'
    with sqlite3.connect(source) as connection:
        assert connection.execute('SELECT event_id, content FROM v2_conversation_events').fetchone() == (event.event_id, 'synthetic apple')
    with sqlite3.connect(target) as connection:
        assert connection.execute('SELECT event_id, content FROM v2_conversation_events').fetchall() == [(event.event_id, 'synthetic apple')]
        assert connection.execute('SELECT event_id FROM v2_conversation_tombstones').fetchall() == [(deleted.event_id,)]
        assert connection.execute('SELECT event_id, content FROM v2_conversation_fts').fetchall() == [(event.event_id, 'synthetic apple')]
    assert [item.event_id for item in SQLiteConversationStore(target).search(owner_user_id='owner', query='apple')] == [event.event_id]


def test_migration_refuses_inconsistent_source_conversation_index(tmp_path):
    source = tmp_path / 'conversations.sqlite3'
    event = SQLiteConversationStore(source).record(owner_user_id='owner', role='user', content='synthetic',
                                                  source='test', source_message_id='fixture')
    with sqlite3.connect(source) as connection:
        connection.execute('DELETE FROM v2_conversation_fts WHERE event_id=?', (event.event_id,))
    target = tmp_path / 'target.sqlite3'
    with pytest.raises(RuntimeError, match='unified_migration_conversation_fts_mismatch'):
        migrate_legacy_databases(target=target, sources=[source], backup_root=tmp_path / 'backups')
    with sqlite3.connect(source) as connection:
        assert connection.execute('SELECT event_id FROM v2_conversation_events').fetchone() == (event.event_id,)
    with sqlite3.connect(target) as connection:
        assert connection.execute('SELECT COUNT(*) FROM v2_conversation_events').fetchone()[0] == 0


def test_migration_refuses_fake_fts_table(tmp_path):
    source = tmp_path / 'conversations.sqlite3'
    with sqlite3.connect(source) as connection:
        connection.execute('CREATE TABLE v2_conversation_fts (event_id TEXT, owner_user_id TEXT, content TEXT)')
        connection.execute("INSERT INTO v2_conversation_fts VALUES ('id', 'owner', 'do not discard')")
    with pytest.raises(RuntimeError, match='unified_migration_conversation_fts_schema_invalid'):
        migrate_legacy_databases(target=tmp_path / 'target.sqlite3', sources=[source], backup_root=tmp_path / 'backups')
    with sqlite3.connect(source) as connection:
        assert connection.execute('SELECT content FROM v2_conversation_fts').fetchone()[0] == 'do not discard'


def test_migration_refuses_resurrection_of_destination_tombstone(tmp_path):
    source = tmp_path / 'conversations.sqlite3'
    event = SQLiteConversationStore(source).record(owner_user_id='owner', role='user', content='synthetic',
                                                  source='test', source_message_id='fixture')
    target = tmp_path / 'target.sqlite3'
    SQLiteConversationStore(target)
    with sqlite3.connect(target) as connection:
        connection.execute('INSERT INTO v2_conversation_tombstones VALUES (?, ?, ?, ?)',
                           (event.event_id, 'owner', 'synthetic', 'digest'))
    with pytest.raises(RuntimeError, match='unified_migration_conversation_tombstone_conflict'):
        migrate_legacy_databases(target=target, sources=[source], backup_root=tmp_path / 'backups')
    with sqlite3.connect(target) as connection:
        assert connection.execute('SELECT COUNT(*) FROM v2_conversation_events').fetchone()[0] == 0
        assert connection.execute('SELECT event_id FROM v2_conversation_tombstones').fetchone() == (event.event_id,)


def test_old_conversation_without_fts_builds_and_verifies_target_index(tmp_path):
    source = tmp_path / 'conversations.sqlite3'
    event = SQLiteConversationStore(source).record(owner_user_id='owner', role='user', content='synthetic pear',
                                                  source='test', source_message_id='fixture')
    with sqlite3.connect(source) as connection:
        connection.execute('DROP TRIGGER v2_conversation_fts_insert')
        connection.execute('DROP TABLE v2_conversation_fts')
    target = tmp_path / 'target.sqlite3'
    assert migrate_legacy_databases(target=target, sources=[source], backup_root=tmp_path / 'backups')['status'] == 'migrated'
    assert [item.event_id for item in SQLiteConversationStore(target).search(owner_user_id='owner', query='pear')] == [event.event_id]


def test_old_conversation_without_fts_refuses_target_index_not_updated(tmp_path, monkeypatch):
    from alphonse.agent_v2 import storage_migration
    source = tmp_path / 'conversations.sqlite3'
    SQLiteConversationStore(source).record(owner_user_id='owner', role='user', content='synthetic pear',
                                           source='test', source_message_id='fixture')
    with sqlite3.connect(source) as connection:
        connection.execute('DROP TRIGGER v2_conversation_fts_insert')
        connection.execute('DROP TABLE v2_conversation_fts')
    target = tmp_path / 'target.sqlite3'
    storage_migration.initialize_unified_schema(target)
    with sqlite3.connect(target) as connection:
        connection.execute('DROP TRIGGER v2_conversation_fts_insert')
    monkeypatch.setattr(storage_migration, 'initialize_unified_schema', lambda _: None)
    with pytest.raises(RuntimeError, match='unified_migration_conversation_fts_mismatch'):
        migrate_legacy_databases(target=target, sources=[source], backup_root=tmp_path / 'backups')
    with sqlite3.connect(target) as connection:
        assert connection.execute('SELECT COUNT(*) FROM v2_conversation_events').fetchone()[0] == 0


def test_explicit_empty_migration_sources_never_scans_default_paths(tmp_path, monkeypatch):
    from alphonse.agent_v2 import storage_migration
    def no_default_scan():
        raise AssertionError('unexpected default source scan')
    monkeypatch.setattr(storage_migration, 'legacy_database_paths', no_default_scan)
    target = tmp_path / 'target.sqlite3'
    result = migrate_legacy_databases(target=target, sources=[], backup_root=tmp_path / 'backups')
    assert result['status'] == 'no_sources'
    assert result['sources'] == []
    assert not target.exists()
    source = tmp_path / 'legacy.sqlite3'
    with sqlite3.connect(source) as connection:
        connection.execute('CREATE TABLE v2_user_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)')
        connection.execute("INSERT INTO v2_user_settings VALUES ('fixture', 'preserved', 'synthetic')")
    assert migrate_legacy_databases(target=target, sources=[source], backup_root=tmp_path / 'backups')['status'] == 'migrated'
    with sqlite3.connect(target) as connection:
        assert connection.execute("SELECT value FROM v2_user_settings WHERE key='fixture'").fetchone()[0] == 'preserved'


@pytest.mark.parametrize('tombstone_in_source', [False, True])
def test_migration_rejects_tombstone_digest_conflict_across_sources(tmp_path, tombstone_in_source):
    source = tmp_path / 'conversations.sqlite3'
    target = tmp_path / 'target.sqlite3'
    source_store, target_store = SQLiteConversationStore(source), SQLiteConversationStore(target)
    event_store = target_store if tombstone_in_source else source_store
    event = event_store.record(owner_user_id='owner', role='user', content='synthetic',
                               source='test', source_message_id='same-origin')
    tombstone_path = source if tombstone_in_source else target
    digest = hashlib.sha256(b'same-origin').hexdigest()
    with sqlite3.connect(tombstone_path) as connection:
        connection.execute('INSERT INTO v2_conversation_tombstones VALUES (?, ?, ?, ?)',
                           ('different-event-id', 'owner', 'synthetic', digest))
    with pytest.raises(RuntimeError, match='unified_migration_conversation_tombstone_conflict'):
        migrate_legacy_databases(target=target, sources=[source], backup_root=tmp_path / 'backups')
    with sqlite3.connect(target) as connection:
        event_ids = [row[0] for row in connection.execute('SELECT event_id FROM v2_conversation_events')]
        assert event_ids == ([event.event_id] if tombstone_in_source else [])


def test_migration_does_not_match_blank_source_ids_to_blank_digest_tombstone(tmp_path):
    source = tmp_path / 'conversations.sqlite3'
    target = tmp_path / 'target.sqlite3'
    event = SQLiteConversationStore(source).record(owner_user_id='owner', role='user', content='synthetic',
                                                  source='test', source_message_id='')
    SQLiteConversationStore(target)
    with sqlite3.connect(target) as connection:
        connection.execute('INSERT INTO v2_conversation_tombstones VALUES (?, ?, ?, ?)',
                           ('different-event-id', 'owner', 'synthetic', hashlib.sha256(b'').hexdigest()))
    assert migrate_legacy_databases(target=target, sources=[source], backup_root=tmp_path / 'backups')['status'] == 'migrated'
    with sqlite3.connect(target) as connection:
        assert connection.execute('SELECT event_id FROM v2_conversation_events').fetchone() == (event.event_id,)


def test_migration_value_comparison_does_not_accept_nocase_conflicts(tmp_path):
    connection = _migration_fixture(tmp_path,
        source_sql='CREATE TABLE v2_fixture (id TEXT PRIMARY KEY, value TEXT)',
        destination_sql='CREATE TABLE v2_fixture (id TEXT PRIMARY KEY COLLATE NOCASE, value TEXT COLLATE NOCASE)',
        source_rows=[('same', 'lower')], target_rows=[('SAME', 'LOWER')])
    with pytest.raises(RuntimeError, match='unified_migration_row_conflict'), connection:
        _copy_table(connection, alias='legacy_0', table='v2_fixture')
    connection.close()


def test_migration_rejects_changed_sqlite_value_type(tmp_path):
    connection = _migration_fixture(tmp_path,
        source_sql='CREATE TABLE v2_fixture (id TEXT PRIMARY KEY, value)',
        destination_sql='CREATE TABLE v2_fixture (id TEXT PRIMARY KEY, value)',
        source_rows=[('same', 1)], target_rows=[('same', 1.0)])
    with pytest.raises(RuntimeError, match='unified_migration_row_conflict'), connection:
        _copy_table(connection, alias='legacy_0', table='v2_fixture')
    connection.close()


def test_checkpoint_omits_ledger_and_resumes_actionable_state(tmp_path) -> None:
    store = SQLiteQuestionStore(tmp_path / "unified.sqlite3")
    ledger = "# Memory Ledger\n" + ("important context " * 100_000)
    task = TaskState(
        task_id="task-one",
        user="alex",
        project_id="project-one",
        goal="Continue the work",
        facts_md="- durable fact",
        conversation_history_md="- User: Continue the work",
        memory_context_md=ledger,
    )
    question = store.create_question(task=task, question="Continue?", kind="yes_no")

    with connect_database(store.db_path) as connection:
        payload = connection.execute(
            "SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task-one'"
        ).fetchone()[0]
    restored = store.load_task_checkpoint("task-one")
    answered = store.route_answer(
        respondent_user_id="alex",
        question_id=question.question_id,
        text="yes",
    )

    assert len(payload) < 20_000
    assert "important context" not in payload
    assert restored is not None
    assert restored.conversation_history_md == EMPTY_MARKDOWN
    assert restored.memory_context_md == EMPTY_MARKDOWN
    assert restored.facts_md == "- durable fact"
    assert answered.resumed_task is not None


def test_retention_removes_only_old_terminal_operational_rows(tmp_path) -> None:
    database = tmp_path / "unified.sqlite3"
    questions = SQLiteQuestionStore(database)
    outbox = SQLiteOutboundStore(database)
    conversations = SQLiteConversationStore(database)
    old = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()

    completed_task = TaskState(task_id="completed", user="alex", goal="Done")
    completed_question = questions.create_question(task=completed_task, question="Done?", kind="yes_no")
    questions.route_answer(respondent_user_id="alex", question_id=completed_question.question_id, text="yes")
    questions.mark_task_checkpoint_terminal("completed", status="done")
    pending_question = questions.create_question(
        task=TaskState(task_id="pending", user="alex", goal="Wait"),
        question="Still pending?",
    )
    delivered = outbox.enqueue(
        address=ChannelAddress("desktop", "tui", "alex", alphonse_user_id="alex"),
        message="Delivered",
    )
    outbox.mark_delivered(delivered.outbox_message_id)
    conversations.record(
        owner_user_id="alex",
        project_id="project",
        role="assistant",
        content="Retain forever",
        source="desktop",
        source_message_id="conversation-one",
    )
    with connect_database(database) as connection:
        connection.execute(
            "UPDATE v2_questions SET updated_at=? WHERE question_id=?",
            (old, completed_question.question_id),
        )
        connection.execute(
            "UPDATE v2_task_checkpoints SET updated_at=? WHERE task_id='completed'",
            (old,),
        )
        connection.execute(
            "UPDATE v2_outbox SET updated_at=? WHERE outbox_message_id=?",
            (old, delivered.outbox_message_id),
        )

    deleted = prune_operational_data(database)

    assert deleted["questions"] == 1
    assert deleted["task_checkpoints"] == 1
    assert deleted["outbound"] == 1
    assert questions.get_question(completed_question.question_id) is None
    assert questions.get_question(pending_question.question_id) is not None
    assert questions.load_task_checkpoint("pending") is not None
    assert len(conversations.list(owner_user_id="alex", project_id="project")) == 1


def test_concurrent_store_writers_share_wal_database(tmp_path) -> None:
    database = tmp_path / "unified.sqlite3"
    SQLiteConversationStore(database)
    SQLiteOutboundStore(database)

    def write(index: int) -> None:
        if index % 2:
            SQLiteConversationStore(database).record(
                owner_user_id=f"user-{index % 3}",
                project_id=f"project-{index % 2}",
                role="user",
                content=f"message {index}",
                source="test",
                source_message_id=f"inbound:{index}",
            )
        else:
            SQLiteOutboundStore(database).enqueue(
                address=ChannelAddress("desktop", "tui", f"user-{index % 3}"),
                message=f"response {index}",
            )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(80)))

    with connect_database(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM v2_conversation_events").fetchone()[0] == 40
        assert connection.execute("SELECT COUNT(*) FROM v2_outbox").fetchone()[0] == 40
