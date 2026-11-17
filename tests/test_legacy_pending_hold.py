"""Disposable prototype checks: hold is not completion or effect reconciliation."""
from datetime import datetime, timezone
import json
import sqlite3

import pytest

from alphonse.agent_v2.automations import EventAutomationStore
from alphonse.agent_v2.core.messages.legacy_pending_hold import (
    hold_legacy_pending, install_legacy_hold_schema, preview_legacy_hold,
)
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.users import V2UserStore

KEY = b'k' * 32
NOW = lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def state(tmp_path):
    path = tmp_path / 'synthetic.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Owner', users_root=tmp_path / 'users')
    other = users.create_user(display_name='Member', role='member')
    SQLiteMessageQueue(path, experimental_claim_fencing=True)
    SQLiteQuestionStore(path)
    SQLiteOutboundStore(path)
    ScheduledTaskStore(path)
    EventAutomationStore(path)
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE v2_trusted_owner_ingress (message_id TEXT PRIMARY KEY)')
        conn.execute('''INSERT INTO v2_inbound_messages
            (message_id,sequence,timestamp,prompt,user_id,metadata_json,queued_at,status)
            VALUES (?,?,?,?,?,?,?,?)''', ('legacy', 1, 'fixture', 'PRIVATE synthetic prompt', owner.user_id,
            json.dumps({'intelligence_engine': 'hierarchical_v3', 'intelligence_schema_version': 3}), 'fixture', 'pending'))
    return path, owner.user_id, other.user_id


def connect(path):
    return sqlite3.connect(path)


def preview(conn, owner):
    return preview_legacy_hold(conn, owner=owner, message_id='legacy', key=KEY)


def hold(conn, owner, token):
    return hold_legacy_pending(conn, owner=owner, message_id='legacy', key=KEY, token=token, clock=NOW)


def test_explicit_hold_preserves_row_and_blocks_old_queue_mutations(state):
    path, owner, _ = state
    with connect(path) as conn:
        install_legacy_hold_schema(conn)
        before = conn.execute('SELECT * FROM v2_inbound_messages').fetchone()
        token = preview(conn, owner)
        assert len(token) == 64 and 'PRIVATE' not in token
        assert hold(conn, owner, token) == 'held'
        assert conn.execute('SELECT * FROM v2_inbound_messages').fetchone() == before
        assert conn.execute('SELECT owner_id FROM v2_legacy_pending_holds').fetchone() == (owner,)
        for sql in ("UPDATE v2_inbound_messages SET status='processing' WHERE message_id='legacy'",
                    "DELETE FROM v2_inbound_messages WHERE message_id='legacy'",
                    "INSERT OR REPLACE INTO v2_inbound_messages SELECT * FROM v2_inbound_messages WHERE message_id='legacy'"):
            with pytest.raises(sqlite3.IntegrityError, match='legacy_pending_hold_requires_reconciliation'):
                conn.execute(sql)
            conn.rollback()


@pytest.mark.parametrize('change', ['prompt', 'claim', 'checkpoint', 'distinct_checkpoint', 'malformed_checkpoint', 'oversize_checkpoint', 'array_checkpoint', 'wrong_message_type', 'duplicate_message_key', 'outbox', 'schedule', 'event', 'owner', 'trigger'])
def test_stale_or_ambiguous_state_refuses_hold(state, change):
    path, owner, _ = state
    with connect(path) as conn:
        install_legacy_hold_schema(conn)
        token = preview(conn, owner)
        if change == 'prompt':
            conn.execute("UPDATE v2_inbound_messages SET prompt='changed' WHERE message_id='legacy'")
        elif change == 'claim':
            conn.execute("UPDATE v2_inbound_messages SET lease_owner='other' WHERE message_id='legacy'")
        elif change == 'checkpoint':
            conn.execute("INSERT INTO v2_task_checkpoints(task_id,task_state_json,status,owner_id,created_at,updated_at) VALUES('legacy','{}','queued',?,'x','x')", (owner,))
        elif change in {'distinct_checkpoint', 'malformed_checkpoint', 'oversize_checkpoint', 'array_checkpoint', 'wrong_message_type', 'duplicate_message_key'}:
            payload = {'distinct_checkpoint': json.dumps({'message_id': 'legacy'}), 'malformed_checkpoint': '{broken',
                       'oversize_checkpoint': json.dumps({'content': 'x' * 65536}), 'array_checkpoint': '[]',
                       'wrong_message_type': '{"message_id":123}',
                       'duplicate_message_key': '{"message_id":"other","message_id":"legacy"}'}[change]
            conn.execute("INSERT INTO v2_task_checkpoints(task_id,task_state_json,status,owner_id,created_at,updated_at) VALUES('different',?,'queued',?,'x','x')", (payload, owner))
        elif change == 'outbox':
            conn.execute("INSERT INTO v2_outbox(outbox_message_id,integration_id,provider_key,channel_target,message,kind,task_id,created_at,updated_at) VALUES('o','x','x','x','x','x','legacy','x','x')")
        elif change == 'schedule':
            conn.execute("INSERT INTO v2_scheduled_task_executions(scheduled_task_id,run_id,status,started_at,queued_message_id) VALUES('x','x','pending','x','legacy')")
        elif change == 'event':
            conn.execute("INSERT INTO v2_automation_executions(execution_id,automation_id,event_key,status,queued_message_id,created_at,updated_at) VALUES('x','x','x','pending','legacy','x','x')")
        elif change == 'owner':
            conn.execute('UPDATE v2_users SET is_active=0 WHERE user_id=?', (owner,))
        else:
            conn.execute('DROP TRIGGER v2_legacy_pending_hold_prevent_update')
        conn.commit()
        with pytest.raises(ValueError):
            hold(conn, owner, token)
        assert conn.execute('SELECT count(*) FROM v2_legacy_pending_holds').fetchone() == (0,)


def test_wrong_actor_key_missing_schema_and_conflicting_guard_fail_closed(state):
    path, owner, other = state
    with connect(path) as conn:
        conn.execute('CREATE TABLE v2_legacy_pending_holds (message_id TEXT)')
        conn.commit()
        with pytest.raises(ValueError, match='schema_unavailable'):
            install_legacy_hold_schema(conn)
        conn.execute('DROP TABLE v2_legacy_pending_holds')
        conn.commit()
        install_legacy_hold_schema(conn)
        with pytest.raises(ValueError):
            preview(conn, other)
        with pytest.raises(ValueError):
            preview_legacy_hold(conn, owner=owner, message_id='legacy', key=b'too short')
        token = preview(conn, owner)
        with pytest.raises(ValueError):
            hold_legacy_pending(conn, owner=owner, message_id='legacy', key=b'z' * 32, token=token, clock=NOW)
        conn.execute('DROP TABLE v2_automation_executions')
        conn.commit()
        with pytest.raises(sqlite3.Error):
            preview(conn, owner)


def test_row_metadata_must_identify_supported_legacy_schema(state):
    path, owner, _ = state
    with connect(path) as conn:
        install_legacy_hold_schema(conn)
        for metadata in ('{}', '{broken', '{"intelligence_engine":"tactical_v2","intelligence_engine":"hierarchical_v3","intelligence_schema_version":3}',
                         '{"intelligence_engine":"hierarchical_v3","intelligence_schema_version":2,"intelligence_schema_version":3}',
                         json.dumps({'intelligence_engine': 'hierarchical_v3', 'intelligence_schema_version': True}),
                         json.dumps({'intelligence_engine': 'single_loop_experimental', 'intelligence_schema_version': 1}),
                         json.dumps({'intelligence_engine': 'hierarchical_v3', 'intelligence_schema_version': 3, 'source': 'unknown'})):
            conn.execute('UPDATE v2_inbound_messages SET metadata_json=? WHERE message_id=?', (metadata, 'legacy'))
            conn.commit()
            with pytest.raises(ValueError):
                preview(conn, owner)


def test_existing_caller_transaction_is_not_rolled_back(state):
    path, owner, _ = state
    with connect(path) as conn:
        install_legacy_hold_schema(conn)
        conn.execute('BEGIN IMMEDIATE')
        conn.execute("UPDATE v2_inbound_messages SET prompt='caller edit' WHERE message_id='legacy'")
        with pytest.raises(ValueError, match='transaction_unavailable'):
            preview(conn, owner)
        assert conn.in_transaction
        assert conn.execute("SELECT prompt FROM v2_inbound_messages WHERE message_id='legacy'").fetchone()[0] == 'caller edit'
        conn.rollback()


def test_oversized_nonprompt_column_refuses_preview(state):
    path, owner, _ = state
    with connect(path) as conn:
        install_legacy_hold_schema(conn)
        conn.execute("UPDATE v2_inbound_messages SET tag=? WHERE message_id='legacy'", ('x' * 65537,))
        conn.commit()
        with pytest.raises(ValueError):
            preview(conn, owner)


@pytest.mark.parametrize('phase', ['install', 'preview', 'hold'])
def test_interrupted_query_rolls_back_and_clears_progress_handler(state, monkeypatch, phase):
    from alphonse.agent_v2.core.messages import legacy_pending_hold as module
    path, owner, _ = state
    with connect(path) as conn:
        if phase != 'install':
            install_legacy_hold_schema(conn)
            token = preview(conn, owner)
        original = module._begin
        def interrupted(connection):
            original(connection)
            connection.set_progress_handler(lambda: 1, 1)
        monkeypatch.setattr(module, '_begin', interrupted)
        with pytest.raises(sqlite3.OperationalError, match='interrupted'):
            if phase == 'install':
                install_legacy_hold_schema(conn)
            elif phase == 'preview':
                preview(conn, owner)
            else:
                hold(conn, owner, token)
        assert not conn.in_transaction
        assert conn.execute("SELECT status FROM v2_inbound_messages WHERE message_id='legacy'").fetchone() == ('pending',)
        if phase != 'install':
            assert conn.execute('SELECT COUNT(*) FROM v2_legacy_pending_holds').fetchone() == (0,)
