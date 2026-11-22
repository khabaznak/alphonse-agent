"""Held drafts stay durable and cannot start a task by themselves."""
import os
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.owner_followups import OwnerFollowups
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def held(tmp_path):
    path = tmp_path / 'fixture.db'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Fixture', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    SQLiteConversationStore(path)
    active = queue.enqueue_trusted_local_owner(prompt='First', peer_uid=os.getuid())
    return queue, owner, active, OwnerFollowups(queue, owner.user_id, os.getuid())


def test_held_in_order_idempotent_and_survives_reopen(held):
    queue, owner, active, store = held
    first, second = '1' * 32, '2' * 32
    assert store.add(active, 'First follow-up', first)['status'] == 'queued'
    assert store.add(active, 'First follow-up', first)['status'] == 'queued'
    store.add(active, 'Second follow-up', second)
    reopened = OwnerFollowups(queue, owner.user_id, os.getuid())
    assert [r['id'] for r in reopened.list()['items']] == [first, second]
    with queue._connect() as conn:
        assert conn.execute('SELECT count(*) FROM v2_inbound_messages').fetchone()[0] == 1
    assert reopened.remove(first)['status'] == 'removed'
    assert reopened.remove(first)['status'] == 'removed'
    with queue._connect() as conn:
        assert conn.execute('SELECT prompt FROM v2_owner_held_followups WHERE id=?', (first,)).fetchone()[0] == ''
    assert store.add(active, 'First follow-up', first)['status'] == 'removed'
    assert [r['id'] for r in reopened.list()['items']] == [second]


def test_untrusted_origin_reuse_and_revocation_fail_closed(held):
    queue, owner, active, store = held
    assert store.add('f' * 32, 'No origin', 'a' * 32)['status'] == 'late'
    store.add(active, 'Original', 'b' * 32)
    with pytest.raises(ValueError):
        store.add(active, 'Changed', 'b' * 32)
    with pytest.raises(PermissionError):
        OwnerFollowups(queue, owner.user_id, os.getuid() + 1)
    with queue._connect() as conn:
        conn.execute('UPDATE v2_users SET is_active=0 WHERE user_id=?', (owner.user_id,))
    with pytest.raises(PermissionError):
        store.list()
    with pytest.raises(PermissionError):
        store.remove('b' * 32)


def test_deleting_linked_turn_requires_removing_held_draft(held):
    queue, owner, active, store = held
    draft = 'c' * 32
    store.add(active, 'Private draft', draft)
    conversations = SQLiteConversationStore(queue.db_path)
    event = conversations.event_for_source_message_id(owner_user_id=owner.user_id,
                                                       source_message_id='inbound:' + active)
    with pytest.raises(ValueError, match='owner_conversation_deletion_requires_reconciliation'):
        conversations.delete(event_id=event.event_id, owner_user_id=owner.user_id)
    store.remove(draft)


def test_unknown_predecessor_never_dispatches_and_remains_removable(held):
    queue, owner, active, store = held
    draft = 'd' * 32
    store.add(active, 'Do not replay', draft)
    # Simulate an unavailable predecessor after restart. A missing queue row
    # alone is not proof of completion or client acknowledgement.
    with queue._connect() as conn:
        conn.execute('DELETE FROM v2_inbound_messages WHERE message_id=?', (active,))
    reopened = OwnerFollowups(queue, owner.user_id, os.getuid())
    assert reopened.release_next() is None
    item = reopened.list()['items'][0]
    assert item['status'] == 'needs_attention' and item['can_remove'] is True
    with queue._connect() as conn:
        assert conn.execute('SELECT count(*) FROM v2_inbound_messages').fetchone()[0] == 0
    assert reopened.remove(draft)['status'] == 'removed'


def test_late_followup_is_not_saved(held):
    queue, _, active, store = held
    with queue._connect() as conn:
        conn.execute('DELETE FROM v2_inbound_messages WHERE message_id=?', (active,))
    assert store.add(active, 'Late', 'e' * 32)['status'] == 'late'
    assert store.list()['items'] == []


def test_deleted_former_owner_has_held_text_removed(held):
    queue, owner, active, store = held
    users = V2UserStore(queue.db_path)
    former = users.create_user(display_name='Former owner')
    store.add(active, 'Retain current owner', 'f' * 32)
    with queue._connect() as conn:
        conn.execute('''INSERT INTO v2_owner_held_followups
            (id,owner_id,peer_uid,active_message_id,prompt,prompt_digest,created_at,status)
            VALUES (?,?,?,?,?,?,?,'queued')''',
            ('a' * 32, former.user_id, os.getuid(), active, 'Former private draft', 'digest', 'now'))
    assert users.delete_user(former.user_id)
    with queue._connect() as conn:
        assert conn.execute('SELECT count(*) FROM v2_owner_held_followups WHERE owner_id=?', (former.user_id,)).fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM v2_owner_held_followups WHERE owner_id=?', (owner.user_id,)).fetchone()[0] == 1


def test_ipc_surfaces_failed_predecessor_and_released_request(held, monkeypatch):
    from alphonse.agent_v2.ipc import V2DaemonServer, PROTOCOL_VERSION
    queue, owner, active, store = held
    store.add(active, 'First', '1' * 32)
    store.add(active, 'Second', '2' * 32)
    with queue._connect() as conn:
        conn.execute("UPDATE v2_owner_held_followups SET status='released' WHERE id=?", ('1' * 32,))
        conn.execute('''INSERT INTO v2_inbound_messages
            (message_id,sequence,timestamp,prompt,user_id,project_id,tag,correlation_id,memory_session_id,metadata_json,queued_at,status)
            SELECT ?,sequence+1,timestamp,prompt,user_id,project_id,tag,correlation_id,memory_session_id,
                   metadata_json,queued_at,status FROM v2_inbound_messages WHERE message_id=?''', ('1' * 32, active))
    users = V2UserStore(queue.db_path)
    daemon = SimpleNamespace(experimental_owner_only=True, runtime=SimpleNamespace(user_store=users, queue=queue),
                             owner_message_status=lambda _: 'failed')
    server = V2DaemonServer(daemon, experimental_owner_ingress=True)
    monkeypatch.setattr(server, '_local_peer_uid', lambda _: os.getuid())
    result = server._owner_followups(None, {'version': PROTOCOL_VERSION,
        'method': 'experimental_owner_followup_list', 'params': {}})
    assert [(item['status'], item['can_remove']) for item in result['items']] == [
        ('needs_attention', False), ('needs_attention', True)]
