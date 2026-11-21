"""Owner-scoped scheduled results, distinct from human receipt."""
import os
from types import SimpleNamespace
from uuid import uuid4

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.simple_loop.owner_schedule_origin import ensure_schedule_origin_schema
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.ipc import PROTOCOL_VERSION, V2DaemonServer
from alphonse.agent_v2.owner_scheduled_delivery import OwnerScheduledDelivery, enqueue_locked
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def fixture(tmp_path):
    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    store = SQLiteConversationStore(path)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    message_id = uuid4().hex
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        ensure_schedule_origin_schema(conn)
        conn.execute('''INSERT INTO v2_owner_schedule_occurrences
            (schedule_id,owner_id,message_id,due_at,payload_sha256,state) VALUES (?,?,?,?,?,'queued')''',
            ('schedule', owner.user_id, message_id, '2026-01-01T00:00:00+00:00', 'a' * 64))
        event = store.record(owner_user_id=owner.user_id, role='assistant', source='scheduled',
            content='The pantry has four cans.', source_message_id='outbound:scheduled:' + message_id,
            connection=conn)
        enqueue_locked(conn, owner_id=owner.user_id, message_id=message_id, event_id=event.event_id, kind='answer')
    return users, owner.user_id, queue, message_id


def test_pending_result_survives_repeated_reads_and_ack_is_owner_scoped(fixture):
    users, owner, _, message_id = fixture
    deliveries = OwnerScheduledDelivery(users.db_path)
    first = deliveries.list_pending(owner)
    assert len(first) == 1 and first[0]['id'] == message_id
    assert first[0]['message'] == 'The pantry has four cans.'
    assert deliveries.list_pending(owner) == first
    with pytest.raises(PermissionError):
        deliveries.list_pending('someone-else')
    with pytest.raises(PermissionError):
        deliveries.acknowledge('someone-else', message_id)
    assert deliveries.acknowledge(owner, message_id) == {'status': 'client_acknowledged'}
    assert deliveries.acknowledge(owner, message_id) == {'status': 'client_acknowledged'}
    assert deliveries.list_pending(owner) == []


def test_delivery_rolls_back_with_scheduled_result(fixture):
    users, owner, queue, _ = fixture
    second = uuid4().hex
    with pytest.raises(RuntimeError):
        with queue._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute('''INSERT INTO v2_owner_schedule_occurrences
                (schedule_id,owner_id,message_id,due_at,payload_sha256,state) VALUES (?,?,?,?,?,'queued')''',
                ('other', owner, second, '2026-01-01T00:00:00+00:00', 'b' * 64))
            event = SQLiteConversationStore._record(conn, event_id=str(uuid4()), owner=owner, project='',
                session='', role='assistant', message='not committed', source='scheduled',
                source_id='outbound:scheduled:' + second, timestamp='2026-01-01T00:00:00+00:00')
            enqueue_locked(conn, owner_id=owner, message_id=second, event_id=event.event_id, kind='answer')
            raise RuntimeError('rollback')
    assert [item['id'] for item in OwnerScheduledDelivery(users.db_path).list_pending(owner)] != [second]


def test_ipc_rejects_forged_actor_and_uses_local_peer(fixture, monkeypatch):
    users, _, queue, message_id = fixture
    daemon = SimpleNamespace(experimental_owner_only=True, runtime=SimpleNamespace(user_store=users, queue=queue))
    server = V2DaemonServer(daemon, experimental_owner_ingress=True)
    req = {'version': PROTOCOL_VERSION, 'method': 'owner_scheduled_results_list', 'params': {}}
    monkeypatch.setattr(server, '_local_peer_uid', lambda _: (_ for _ in ()).throw(PermissionError('denied')))
    with pytest.raises(PermissionError):
        server._owner_scheduled_delivery(None, req)
    monkeypatch.setattr(server, '_local_peer_uid', lambda _: os.getuid())
    with pytest.raises(ValueError, match='payload_denied'):
        server._owner_scheduled_delivery(None, {**req, 'params': {'actor_user_id': 'forged'}})
    assert server._owner_scheduled_delivery(None, req)['results'][0]['id'] == message_id
    ack = {'version': PROTOCOL_VERSION, 'method': 'owner_scheduled_result_ack', 'params': {'id': message_id}}
    assert server._owner_scheduled_delivery(None, ack) == {'status': 'client_acknowledged'}
