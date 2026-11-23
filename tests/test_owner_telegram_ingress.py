import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_ingress import enqueue_telegram_owner, verified_telegram_origin_locked, telegram_owner_claim_transaction
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable
from alphonse.agent_v2.core.messages.queue import MessageSelector
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def fixture(tmp_path):
    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    users.bind_address(user_id=owner.user_id, integration_id='telegram-test', provider_key='telegram', provider_user_id='123')
    SQLiteConversationStore(path)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    with queue._connect() as conn:
        conn.execute('''CREATE TABLE v2_integrations (integration_id TEXT PRIMARY KEY,provider_key TEXT,enabled INTEGER,
            config_json TEXT,credential_generation INTEGER,revocation_pending INTEGER,credential_deletion_pending INTEGER,updated_at TEXT)''')
        conn.execute('INSERT INTO v2_integrations VALUES (?,?,?,?,?,?,?,?)', ('telegram-test','telegram',1,'{}',1,0,0,'r1'))
    return queue, owner


def update(text='hello', *, sender=123, chat=123, kind='private', update_id=42):
    return {'update_id': update_id, 'message': {'message_id': 9, 'text': text,
        'from': {'id': sender}, 'chat': {'id': chat, 'type': kind}}}


def test_atomic_origin_is_distinct_from_desktop_and_deduplicates(fixture):
    queue, owner = fixture
    mid = enqueue_telegram_owner(queue, integration_id='telegram-test', update=update())
    assert enqueue_telegram_owner(queue, integration_id='telegram-test', update=update()) == mid
    with pytest.raises(ValueError, match='duplicate_conflict'):
        enqueue_telegram_owner(queue, integration_id='telegram-test', update=update('changed'))
    with queue._connect() as conn:
        origin = verified_telegram_origin_locked(conn, mid)
        assert origin['owner_id'] == owner.user_id
        assert origin['chat_id'] == '123'
        assert conn.execute('SELECT count(*) FROM v2_inbound_messages').fetchone()[0] == 1
        assert conn.execute("SELECT source FROM v2_conversation_events WHERE source_message_id=?", (f'inbound:{mid}',)).fetchone()[0] == 'telegram'
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='v2_trusted_owner_ingress'").fetchone()


@pytest.mark.parametrize('event', [update(sender=456), update(chat=456), update(kind='group')])
def test_non_owner_and_group_cannot_enter(fixture, event):
    queue, _ = fixture
    with pytest.raises((PermissionError, ValueError)):
        enqueue_telegram_owner(queue, integration_id='telegram-test', update=event)


@pytest.mark.parametrize('sql', ["UPDATE v2_inbound_messages SET prompt='tampered'",
    "UPDATE v2_user_addresses SET is_active=0", "UPDATE v2_integrations SET enabled=0",
    "UPDATE v2_integrations SET credential_generation=2", "UPDATE v2_conversation_events SET content='tampered'",
    "UPDATE v2_user_addresses SET updated_at='revoked-and-restored'", "UPDATE v2_integrations SET updated_at='disabled-and-restored'"])
def test_origin_revoked_or_changed_is_not_accepted(fixture, sql):
    queue, _ = fixture
    mid = enqueue_telegram_owner(queue, integration_id='telegram-test', update=update())
    with queue._connect() as conn:
        assert verified_telegram_origin_locked(conn, mid)
        conn.execute(sql)
        assert verified_telegram_origin_locked(conn, mid) is None


def test_telegram_claim_reaches_checkpoint_without_desktop_impersonation(fixture):
    from datetime import datetime, timezone
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_atomic_checkpoint import OwnerAtomicCheckpoint
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_processor import authorize_trusted_owner_claim
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_direct_claim import verify_direct_owner_claim
    from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_checkpoint import ExperimentalCheckpointStore
    from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable
    from alphonse.agent_v2.core.questions import SQLiteQuestionStore
    queue, _ = fixture
    mid = enqueue_telegram_owner(queue, integration_id='telegram-test', update=update())
    acquired = queue.claim_next()
    task = TaskState.from_message(acquired.message, message_id=mid)
    task.task_id = mid
    context = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt)
    assert not queue.verify_trusted_local_owner_claim(acquired.claim_receipt, task)
    assert authorize_trusted_owner_claim(task, context)
    assert verify_direct_owner_claim(queue, acquired.claim_receipt, task)
    atomic = OwnerAtomicCheckpoint(queue, ExperimentalCheckpointStore(SQLiteQuestionStore(queue.db_path)),
                                   task, acquired.claim_receipt, clock=lambda: datetime.now(timezone.utc))
    assert atomic.telegram_origin and not atomic.scheduled_origin
    with atomic._transaction() as conn:
        assert verified_telegram_origin_locked(conn, mid)['owner_id'] == task.user
    with queue._connect() as conn:
        conn.execute('UPDATE v2_user_addresses SET is_active=0')
    assert not authorize_trusted_owner_claim(task, context)
    assert not verify_direct_owner_claim(queue, acquired.claim_receipt, task)
    with pytest.raises(ClaimFenceUnavailable):
        with atomic._transaction():
            pass


def test_fenced_telegram_claim_is_distinct_and_rechecks_revocation(fixture):
    queue, _ = fixture
    mid = enqueue_telegram_owner(queue, integration_id='telegram-test', update=update())
    acquired = queue.claim_next(selector=MessageSelector(message_id=mid), lease_owner='synthetic', lease_seconds=180)
    task = TaskState.from_message(acquired.message, message_id=mid)
    task.task_id = mid
    assert not queue.verify_trusted_local_owner_claim(acquired.claim_receipt, task)
    with telegram_owner_claim_transaction(queue, acquired.claim_receipt, task) as conn:
        assert verified_telegram_origin_locked(conn, mid)['message_id'] == mid
    with queue._connect() as conn:
        conn.execute('UPDATE v2_user_addresses SET is_active=0')
    with pytest.raises(ClaimFenceUnavailable):
        with telegram_owner_claim_transaction(queue, acquired.claim_receipt, task):
            pass
