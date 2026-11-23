"""Disposable checks for the unregistered owner knowledge snapshot."""
import json
import os
from dataclasses import replace

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.simple_loop.owner_knowledge_snapshot import (
    snapshot_owner_knowledge, snapshot_owner_knowledge_locked, matches_owner_knowledge_locked,
)
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.memory.contracts import AccessScope, conversation_item
from alphonse.agent_v2.core.memory.knowledge import KnowledgeRecord, KnowledgeStore, _source_snapshot_digest
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def fixture(tmp_path):
    path = tmp_path / 'history.db'
    owner = V2UserStore(path).onboard(display_name='Fixture', users_root=tmp_path / 'users')
    conversations = SQLiteConversationStore(path)
    KnowledgeStore(path)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    source = conversations.record(owner_user_id=owner.user_id, role='user', content='Apples are red',
                                  source='fixture', source_message_id='earlier')
    message_id = queue.enqueue_trusted_local_owner(prompt='current request', peer_uid=os.getuid())
    acquired = queue.claim_next()
    task = TaskState.from_message(acquired.message, message_id=message_id)
    task.task_id = message_id
    item = conversation_item(source, owner.user_id)
    record = KnowledgeRecord(memory_id='m00', content='Apples are red', access=AccessScope(owner.user_id),
                             sources=(item.source,), recorded_at=source.created_at, kind='fact', subjects=('apples',),
                             claim_key='apple', basis='user_report', quotes=('Apples are red',),
                             source_binding_version=1, source_snapshot_digests=(_source_snapshot_digest(item),))
    def insert(r):
        with queue._connect() as conn:
            conn.execute('INSERT INTO memory_knowledge VALUES (?,?,?,?,?,?)',
                         (r.memory_id, owner.user_id, r.claim_key, json.dumps(r.to_dict()), r.status, '[]'))
    def snapshot(cancelled=False):
        return snapshot_owner_knowledge(queue, acquired.claim_receipt, task, cancellation_checker=lambda: cancelled)
    snapshot.receipt = acquired.claim_receipt
    return queue, task, source, record, insert, snapshot


def test_valid_snapshot_and_changed_source(fixture):
    queue, _, source, record, insert, snapshot = fixture
    insert(record)
    result = snapshot()
    assert result.status == 'available'
    assert [item.content for item in result.items] == ['Apples are red']
    with queue._connect() as conn:
        conn.execute('UPDATE v2_conversation_events SET content=? WHERE event_id=?', ('Apples are green', source.event_id))
    assert snapshot().status == 'incomplete'


def test_telegram_owner_snapshot_respects_live_sender_and_sources(tmp_path):
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_ingress import enqueue_telegram_owner
    from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
    path = tmp_path / 'telegram.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Fixture', users_root=tmp_path / 'users')
    users.bind_address(user_id=owner.user_id, integration_id='telegram-fixture', provider_key='telegram', provider_user_id='123')
    SQLiteIntegrationStore(path).upsert(integration_id='telegram-fixture', provider_key='telegram', display_name='Telegram', enabled=True)
    conversations = SQLiteConversationStore(path)
    KnowledgeStore(path)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    source = conversations.record(owner_user_id=owner.user_id, role='user', content='Apples are red',
                                  source='fixture', source_message_id='earlier')
    mid = enqueue_telegram_owner(queue, integration_id='telegram-fixture', update={'update_id': 42,
        'message': {'message_id': 9, 'text': 'What color are apples?', 'from': {'id': 123},
                    'chat': {'id': 123, 'type': 'private'}}})
    acquired = queue.claim_next()
    task = TaskState.from_message(acquired.message, message_id=mid); task.task_id = mid
    item = conversation_item(source, owner.user_id)
    record = KnowledgeRecord(memory_id='m00', content='Apples are red', access=AccessScope(owner.user_id),
        sources=(item.source,), recorded_at=source.created_at, kind='fact', subjects=('apples',),
        claim_key='apple', basis='user_report', quotes=('Apples are red',), source_binding_version=1,
        source_snapshot_digests=(_source_snapshot_digest(item),))
    with queue._connect() as conn:
        conn.execute('INSERT INTO memory_knowledge VALUES (?,?,?,?,?,?)',
                     (record.memory_id, owner.user_id, record.claim_key, json.dumps(record.to_dict()), record.status, '[]'))
    snapshot = lambda: snapshot_owner_knowledge(queue, acquired.claim_receipt, task, cancellation_checker=lambda: False)
    assert [item.content for item in snapshot().items] == ['Apples are red']
    with queue._connect() as conn:
        conn.execute('UPDATE v2_user_addresses SET is_active=0')
    assert snapshot().status == 'unavailable'


def test_cached_descendants_cannot_bypass_depth_limit(fixture):
    _, _, _, record, insert, snapshot = fixture
    for i in range(17):
        insert(replace(record, memory_id=f'm{i:02}', dependencies=(f'm{i-1:02}',) if i else ()))
    assert snapshot().status == 'incomplete'


@pytest.mark.parametrize('field', ['payload', 'conflicts'])
def test_oversized_knowledge_fields_fail_closed(fixture, field):
    queue, _, _, record, insert, snapshot = fixture
    insert(record)
    with queue._connect() as conn:
        conn.execute(f'UPDATE memory_knowledge SET {field}=?', ('x' * 65537,))
    assert snapshot().status == 'incomplete'


def test_oversized_conversation_fields_fail_closed(fixture):
    queue, _, source, record, insert, snapshot = fixture
    insert(record)
    with queue._connect() as conn:
        conn.execute('UPDATE v2_conversation_events SET content=? WHERE event_id=?', ('x' * 65537, source.event_id))
    assert snapshot().status == 'incomplete'


def test_unbound_and_non_user_source_are_incomplete(fixture):
    queue, _, source, record, insert, snapshot = fixture
    insert(replace(record, source_binding_version=0, source_snapshot_digests=()))
    assert snapshot().status == 'incomplete'
    with queue._connect() as conn:
        conn.execute('DELETE FROM memory_knowledge')
        conn.execute("UPDATE v2_conversation_events SET role='assistant' WHERE event_id=?", (source.event_id,))
    insert(record)
    assert snapshot().status == 'incomplete'


def test_missing_inbound_and_cancellation_unavailable(fixture):
    queue, task, _, _, _, snapshot = fixture
    with queue._connect() as conn:
        conn.execute('DELETE FROM v2_conversation_events WHERE source_message_id=?', (f'inbound:{task.message_id}',))
    assert snapshot().status == 'unavailable'


def test_current_request_and_dependent_knowledge_are_excluded(fixture):
    queue, task, _, record, insert, snapshot = fixture
    from alphonse.agent_v2.conversations import _event
    with queue._connect() as conn:
        row = conn.execute('SELECT * FROM v2_conversation_events WHERE source_message_id=?',
                           (f'inbound:{task.message_id}',)).fetchone()
    item = conversation_item(_event(row), task.user)
    current = replace(record, memory_id='current', content='current request', sources=(item.source,),
                      quotes=('current request',), source_snapshot_digests=(_source_snapshot_digest(item),))
    insert(current)
    insert(replace(record, memory_id='dependent', dependencies=('current',)))
    result = snapshot()
    assert result.status == 'available'
    assert result.items == ()


def test_cancelled_claim_yields_no_content(fixture):
    _, _, _, record, insert, snapshot = fixture
    insert(record)
    result = snapshot(cancelled=True)
    assert result.status == 'unavailable'
    assert result.items == ()


def test_locked_snapshot_shares_claim_transaction_and_leaves_it_writable(fixture):
    queue, task, source, record, insert, snapshot = fixture
    insert(record)
    with queue.trusted_local_owner_claim_transaction(snapshot.receipt, task) as conn:
        before = snapshot_owner_knowledge_locked(conn, task, cancellation_checker=lambda: False)
        assert before.status == 'available'
        conn.execute('UPDATE v2_conversation_events SET content=? WHERE event_id=?',
                     ('Apples are green', source.event_id))
        after = snapshot_owner_knowledge_locked(conn, task, cancellation_checker=lambda: False)
        assert after.status == 'incomplete'
        assert after.items == ()
    assert snapshot().status == 'incomplete'


def test_locked_snapshot_requires_transaction_and_rechecks_cancellation(fixture):
    queue, task, _, record, insert, snapshot = fixture
    insert(record)
    with queue._connect() as conn:
        assert snapshot_owner_knowledge_locked(conn, task, cancellation_checker=lambda: False).status == 'unavailable'
    with queue.trusted_local_owner_claim_transaction(snapshot.receipt, task) as conn:
        result = snapshot_owner_knowledge_locked(conn, task, cancellation_checker=lambda: True)
        assert result.status == 'unavailable'
        assert result.items == ()


def test_locked_binding_detects_rebound_sources_with_unchanged_public_claim(fixture):
    from alphonse.agent_v2.conversations import _event

    queue, task, source, record, insert, snapshot = fixture
    insert(record)
    previous = snapshot()
    with queue.trusted_local_owner_claim_transaction(snapshot.receipt, task) as conn:
        assert matches_owner_knowledge_locked(conn, task, previous.digest, cancellation_checker=lambda: False)
        conn.execute('UPDATE v2_conversation_events SET content=? WHERE event_id=?',
                     ('Apples are red, and taste sweet.', source.event_id))
        row = conn.execute('SELECT * FROM v2_conversation_events WHERE event_id=?', (source.event_id,)).fetchone()
        changed = conversation_item(_event(row), task.user)
        rebound = replace(record, source_snapshot_digests=(_source_snapshot_digest(changed),))
        conn.execute('UPDATE memory_knowledge SET payload=? WHERE memory_id=?',
                     (json.dumps(rebound.to_dict()), record.memory_id))
        current = snapshot_owner_knowledge_locked(conn, task, cancellation_checker=lambda: False)
        assert current.items == previous.items
        assert not matches_owner_knowledge_locked(conn, task, previous.digest, cancellation_checker=lambda: False)
        assert matches_owner_knowledge_locked(conn, task, current.digest, cancellation_checker=lambda: False)
        assert not matches_owner_knowledge_locked(conn, task, current.digest, cancellation_checker=lambda: True)
