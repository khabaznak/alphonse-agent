"""A migration must never turn an unknown delivery state into a pending send."""
import sqlite3

import pytest

from alphonse.agent_v2.core.io.channels import ChannelAddress
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore, _migrate_status_check


def _legacy_database(path, statuses, *, permits_unconfirmed=False, permits_future=False,
                     permits_null_message=False, ddl_transform=None):
    store = SQLiteOutboundStore(path)
    address = ChannelAddress('desktop', 'desktop', 'alice', alphonse_user_id='alice')
    for index in range(len(statuses)):
        store.enqueue(address=address, message=f'synthetic-{index}', idempotency_key=f'item-{index}')
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        ddl = conn.execute("SELECT sql FROM sqlite_master WHERE name='v2_outbox' AND type='table'").fetchone()['sql']
        options = "'pending', 'claimed', 'retry_wait', 'delivered', 'failed', 'unconfirmed'"
        allowed = "'pending', 'claimed', 'retry_wait', 'delivered', 'failed'"
        if permits_future or 'future_hold' in statuses:
            allowed += ", 'future_hold'"
        if permits_unconfirmed:
            allowed += ", 'unconfirmed'"
        assert options in ddl
        conn.execute('ALTER TABLE v2_outbox RENAME TO source_outbox')
        legacy_ddl = ddl.replace(options, allowed)
        if permits_null_message:
            assert 'message TEXT NOT NULL' in legacy_ddl
            legacy_ddl = legacy_ddl.replace('message TEXT NOT NULL', 'message TEXT')
        if ddl_transform:
            legacy_ddl = ddl_transform(legacy_ddl)
        conn.execute(legacy_ddl)
        conn.execute('INSERT INTO v2_outbox SELECT * FROM source_outbox')
        conn.execute('DROP TABLE source_outbox')
        conn.execute('CREATE INDEX idx_v2_outbox_consumer ON v2_outbox(integration_id, channel_target, status, created_at)')
        conn.execute('CREATE INDEX idx_v2_outbox_question ON v2_outbox(question_id, status)')
        conn.execute('CREATE INDEX idx_v2_outbox_project ON v2_outbox(audience_user_id, project_id, status, created_at)')
        ids = [row[0] for row in conn.execute('SELECT outbox_message_id FROM v2_outbox ORDER BY created_at')]
        for outbox_id, status in zip(ids, statuses):
            conn.execute('UPDATE v2_outbox SET status=? WHERE outbox_message_id=?', (status, outbox_id))


def _snapshot(conn):
    rows = [tuple(row) for row in conn.execute('SELECT * FROM v2_outbox ORDER BY outbox_message_id')]
    indexes = [tuple(row) for row in conn.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='v2_outbox' AND sql IS NOT NULL ORDER BY name")]
    return rows, indexes


@pytest.mark.parametrize('permits_unconfirmed', [False, True])
def test_unknown_state_is_not_migrated_or_dispatched(tmp_path, permits_unconfirmed):
    path = tmp_path / 'legacy.sqlite'
    _legacy_database(path, ['pending', 'future_hold'], permits_unconfirmed=permits_unconfirmed)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        before = _snapshot(conn)
        with pytest.raises(ValueError, match='^outbox_status_unsupported$'):
            _migrate_status_check(conn)
        assert _snapshot(conn) == before
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE name='v2_outbox_legacy'").fetchone()[0] == 0
    # Initialization also refuses the store without exposing private row text.
    with pytest.raises(ValueError, match='^outbox_status_unsupported$'):
        SQLiteOutboundStore(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM v2_outbox WHERE status='future_hold'").fetchone()[0] == 1


def test_supported_legacy_states_and_required_indexes_survive_migration(tmp_path):
    path = tmp_path / 'legacy.sqlite'
    statuses = ['pending', 'claimed', 'retry_wait', 'delivered', 'failed']
    _legacy_database(path, statuses)
    with sqlite3.connect(path) as conn:
        before = sorted(conn.execute('SELECT outbox_message_id, status, message FROM v2_outbox'))
    SQLiteOutboundStore(path)
    with sqlite3.connect(path) as conn:
        assert sorted(conn.execute('SELECT outbox_message_id, status, message FROM v2_outbox')) == before
        indexes = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='v2_outbox'")}
        assert {'idx_v2_outbox_consumer', 'idx_v2_outbox_question', 'idx_v2_outbox_project'} <= indexes


def test_existing_unconfirmed_outcome_remains_unconfirmed(tmp_path):
    path = tmp_path / 'current.sqlite'
    _legacy_database(path, ['unconfirmed'], permits_unconfirmed=True)
    SQLiteOutboundStore(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT status FROM v2_outbox').fetchone()[0] == 'unconfirmed'


def test_future_status_schema_is_rejected_even_when_no_future_rows_exist(tmp_path):
    path = tmp_path / 'future.sqlite'
    _legacy_database(path, ['pending'], permits_unconfirmed=True, permits_future=True)
    with sqlite3.connect(path) as conn:
        before = _snapshot(conn)
        before_sql = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='v2_outbox'").fetchone()[0]
    with pytest.raises(ValueError, match='^outbox_status_schema_unsupported$'):
        SQLiteOutboundStore(path)
    with sqlite3.connect(path) as conn:
        assert _snapshot(conn) == before
        assert conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='v2_outbox'").fetchone()[0] == before_sql


def test_migration_failure_rolls_back_ddl_data_and_indexes(tmp_path, monkeypatch):
    import alphonse.agent_v2.core.io.outbox as outbox

    path = tmp_path / 'invalid-legacy-row.sqlite'
    _legacy_database(path, ['pending'])
    with sqlite3.connect(path) as conn:
        before = _snapshot(conn)
        before_schema = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='v2_outbox'").fetchone()[0]
    original = outbox._migrate_status_check
    def fail_after_rebuild(conn):
        original(conn)
        raise RuntimeError('synthetic_failure_after_rebuild')
    monkeypatch.setattr(outbox, '_migrate_status_check', fail_after_rebuild)
    with pytest.raises(RuntimeError, match='synthetic_failure_after_rebuild'):
        SQLiteOutboundStore(path)
    with sqlite3.connect(path) as conn:
        assert _snapshot(conn) == before
        assert conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='v2_outbox'").fetchone()[0] == before_schema
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE name='v2_outbox_legacy'").fetchone()[0] == 0


def test_preflight_and_migration_hold_one_write_lock(tmp_path, monkeypatch):
    from threading import Event, Thread
    import alphonse.agent_v2.core.io.outbox as outbox

    path = tmp_path / 'concurrent.sqlite'
    _legacy_database(path, ['pending'], permits_unconfirmed=True, permits_future=True)
    entered, resume = Event(), Event()
    original = outbox._reject_unsupported_statuses

    def paused_preflight(conn):
        original(conn)
        entered.set()
        assert resume.wait(5)

    monkeypatch.setattr(outbox, '_reject_unsupported_statuses', paused_preflight)
    result = []

    def initialize():
        try:
            SQLiteOutboundStore(path)
        except Exception as exc:
            result.append(exc)

    worker = Thread(target=initialize)
    worker.start()
    try:
        assert entered.wait(5)
        with sqlite3.connect(path, timeout=0) as other:
            with pytest.raises(sqlite3.OperationalError, match='locked'):
                other.execute("UPDATE v2_outbox SET status='future_hold'")
    finally:
        resume.set()
        worker.join(5)
    assert not worker.is_alive()
    assert len(result) == 1 and str(result[0]) == 'outbox_status_schema_unsupported'
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT status FROM v2_outbox').fetchone()[0] == 'pending'


@pytest.mark.parametrize('extension', ['index', 'index_collation', 'index_desc', 'trigger', 'column', 'foreign_key',
                                       'view', 'other_trigger', 'generated', 'collation', 'nullable', 'default', 'unique', 'type'])
def test_unrecognized_legacy_schema_objects_are_not_dropped(tmp_path, extension):
    path = tmp_path / 'extended.sqlite'
    transforms = {
        'collation': lambda ddl: ddl.replace('message TEXT NOT NULL', 'message TEXT COLLATE NOCASE NOT NULL'),
        'nullable': lambda ddl: ddl.replace('message TEXT NOT NULL', 'message TEXT'),
        'type': lambda ddl: ddl.replace('message TEXT NOT NULL', 'message ANY NOT NULL'),
        'default': lambda ddl: ddl.replace("metadata_json TEXT NOT NULL DEFAULT '{}'", "metadata_json TEXT NOT NULL DEFAULT 'changed'"),
        'unique': lambda ddl: ddl.replace('CHECK (status IN', 'UNIQUE(message), CHECK (status IN'),
    }
    _legacy_database(path, ['pending'], ddl_transform=transforms.get(extension))
    with sqlite3.connect(path) as conn:
        if extension == 'index':
            conn.execute('CREATE INDEX extra_outbox_index ON v2_outbox(message)')
        elif extension in {'index_collation', 'index_desc'}:
            conn.execute('DROP INDEX idx_v2_outbox_question')
            qualifier = 'COLLATE NOCASE' if extension == 'index_collation' else 'DESC'
            conn.execute(f'CREATE INDEX idx_v2_outbox_question ON v2_outbox(question_id {qualifier}, status)')
        elif extension == 'trigger':
            conn.execute('CREATE TRIGGER extra_outbox_trigger AFTER UPDATE ON v2_outbox BEGIN SELECT 1; END')
        elif extension == 'column':
            conn.execute("ALTER TABLE v2_outbox ADD COLUMN extra_data TEXT DEFAULT 'keep'")
        elif extension == 'foreign_key':
            conn.execute('CREATE TABLE dependent(id TEXT, outbound TEXT REFERENCES v2_outbox(outbox_message_id))')
        elif extension == 'view':
            conn.execute('CREATE VIEW audit_view AS SELECT status FROM v2_outbox')
        elif extension == 'other_trigger':
            conn.execute('CREATE TABLE source(id TEXT)')
            conn.execute("CREATE TRIGGER source_trigger AFTER INSERT ON source BEGIN UPDATE v2_outbox SET status='failed'; END")
        elif extension == 'generated':
            conn.execute('ALTER TABLE v2_outbox ADD COLUMN derived TEXT GENERATED ALWAYS AS (message) VIRTUAL')
        before = _snapshot(conn)
        schema = list(conn.execute("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name"))
    with pytest.raises(ValueError, match='^outbox_rebuild_schema_unsupported$'):
        SQLiteOutboundStore(path)
    with sqlite3.connect(path) as conn:
        assert _snapshot(conn) == before
        assert list(conn.execute("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name")) == schema


def test_oldest_recorded_status_schema_and_missing_columns_upgrade(tmp_path):
    path = tmp_path / 'oldest.sqlite'
    # The checked-in initial outbox schema had this four-status CHECK and did
    # not yet include project, retry or lease columns.
    with sqlite3.connect(path) as conn:
        conn.executescript("""
        CREATE TABLE v2_outbox (
          outbox_message_id TEXT PRIMARY KEY, integration_id TEXT NOT NULL,
          provider_key TEXT NOT NULL, channel_target TEXT NOT NULL, message TEXT NOT NULL,
          kind TEXT NOT NULL, audience_user_id TEXT NOT NULL DEFAULT '', correlation_id TEXT NOT NULL DEFAULT '',
          status TEXT NOT NULL DEFAULT 'pending', provider_message_id TEXT NOT NULL DEFAULT '',
          reply_to_provider_message_id TEXT NOT NULL DEFAULT '', task_id TEXT NOT NULL DEFAULT '',
          question_id TEXT NOT NULL DEFAULT '', metadata_json TEXT NOT NULL DEFAULT '{}',
          claimed_at TEXT NOT NULL DEFAULT '', delivered_at TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
          CHECK (status IN ('pending', 'claimed', 'delivered', 'failed'))
        ) STRICT;
        CREATE INDEX idx_v2_outbox_consumer ON v2_outbox (integration_id, channel_target, status, created_at);
        CREATE INDEX idx_v2_outbox_question ON v2_outbox (question_id, status);
        INSERT INTO v2_outbox(outbox_message_id,integration_id,provider_key,channel_target,message,kind,status,created_at,updated_at)
        VALUES ('old', 'desktop', 'desktop', 'alice', 'synthetic', 'response', 'delivered', 't', 't');
        """)
    store = SQLiteOutboundStore(path)
    message = store.get('old')
    assert message.status == 'delivered'
    assert message.message == 'synthetic'
    with sqlite3.connect(path) as conn:
        indexes = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='v2_outbox'")}
        assert {'idx_v2_outbox_consumer', 'idx_v2_outbox_question', 'idx_v2_outbox_project'} <= indexes


def test_unsupported_sparse_legacy_store_is_not_altered_before_refusal(tmp_path):
    path = tmp_path / 'sparse.sqlite'
    with sqlite3.connect(path) as conn:
        # Deliberately lacks the newer columns and built-in indexes. An unsafe
        # preflight order could add them or update data before detecting status.
        conn.execute('CREATE TABLE v2_outbox(outbox_message_id TEXT PRIMARY KEY, status TEXT, message TEXT)')
        conn.execute('CREATE INDEX private_index ON v2_outbox(message)')
        conn.execute("INSERT INTO v2_outbox VALUES ('item', 'future_hold', 'synthetic private data')")
    with sqlite3.connect(path) as conn:
        before = _snapshot(conn)
        before_schema = [tuple(row) for row in conn.execute('PRAGMA table_info(v2_outbox)')]
        before_sql = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='v2_outbox'").fetchone()[0]
        all_schema = list(conn.execute("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name"))
        journal = conn.execute('PRAGMA journal_mode').fetchone()[0]
    with pytest.raises(ValueError, match='^outbox_status_unsupported$') as error:
        SQLiteOutboundStore(path)
    assert 'synthetic private data' not in str(error.value)
    with sqlite3.connect(path) as conn:
        assert _snapshot(conn) == before
        assert [tuple(row) for row in conn.execute('PRAGMA table_info(v2_outbox)')] == before_schema
        assert conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='v2_outbox'").fetchone()[0] == before_sql
        assert list(conn.execute("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name")) == all_schema
        assert conn.execute('PRAGMA journal_mode').fetchone()[0] == journal


def test_existing_store_without_status_is_rejected_without_mutation(tmp_path):
    path = tmp_path / 'invalid.sqlite'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE v2_outbox(outbox_message_id TEXT PRIMARY KEY, message TEXT)')
        conn.execute("INSERT INTO v2_outbox VALUES ('item', 'synthetic private data')")
    with pytest.raises(ValueError, match='^outbox_status_schema_invalid$'):
        SQLiteOutboundStore(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT * FROM v2_outbox').fetchall() == [('item', 'synthetic private data')]
        assert [row[1] for row in conn.execute('PRAGMA table_info(v2_outbox)')] == ['outbox_message_id', 'message']


@pytest.mark.parametrize('status_definition', [
    "status TEXT /* CHECK(status IN ('pending','claimed','retry_wait','delivered','failed','unconfirmed')) */",
    "status TEXT, extra TEXT DEFAULT 'CHECK(status IN (''pending'',''claimed'',''retry_wait'',''delivered'',''failed'',''unconfirmed''))'",
    '''status TEXT, "CHECK(status IN ('pending','claimed','retry_wait','delivered','failed','unconfirmed'))" TEXT''',
])
def test_check_text_in_comment_or_literal_is_not_a_constraint(tmp_path, status_definition):
    path = tmp_path / 'spoof.sqlite'
    with sqlite3.connect(path) as conn:
        conn.execute(f'CREATE TABLE v2_outbox(outbox_message_id TEXT PRIMARY KEY, {status_definition})')
        conn.execute("INSERT INTO v2_outbox(outbox_message_id,status) VALUES ('item','pending')")
        before = list(conn.execute("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name"))
    with pytest.raises(ValueError, match='^outbox_status_schema_unsupported$'):
        SQLiteOutboundStore(path)
    with sqlite3.connect(path) as conn:
        assert list(conn.execute("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name")) == before
        assert conn.execute('SELECT status FROM v2_outbox').fetchone()[0] == 'pending'


def test_schema_preflight_closes_file_connection_on_failure(tmp_path, monkeypatch):
    import alphonse.agent_v2.core.io.outbox as outbox
    from alphonse.agent_v2.database import AlphonseSQLiteConnection

    path = tmp_path / 'unsupported.sqlite'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE v2_outbox(outbox_message_id TEXT PRIMARY KEY, status TEXT)')
        conn.execute("INSERT INTO v2_outbox VALUES ('item', 'future_hold')")
    closed = []
    class TrackedConnection(AlphonseSQLiteConnection):
        def close(self):
            closed.append(True)
            super().close()
    monkeypatch.setattr(outbox, 'AlphonseSQLiteConnection', TrackedConnection)
    with pytest.raises(ValueError, match='^outbox_status_unsupported$'):
        SQLiteOutboundStore(path)
    assert closed == [True]


def test_successful_schema_init_keeps_shared_connection_settings(tmp_path):
    from alphonse.agent_v2.database import AlphonseSQLiteConnection, DEFAULT_BUSY_TIMEOUT_MS

    store = SQLiteOutboundStore(tmp_path / 'runtime.sqlite')
    with store._connect() as conn:
        assert isinstance(conn, AlphonseSQLiteConnection)
        assert conn.row_factory is sqlite3.Row
        assert conn.execute('PRAGMA foreign_keys').fetchone()[0] == 1
        assert conn.execute('PRAGMA busy_timeout').fetchone()[0] == DEFAULT_BUSY_TIMEOUT_MS
        assert conn.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
        assert conn.execute('PRAGMA synchronous').fetchone()[0] == 1
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='v2_schema_migrations'").fetchone() is not None
