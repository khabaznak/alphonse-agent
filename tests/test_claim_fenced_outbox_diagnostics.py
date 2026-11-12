"""Disposable transaction experiments, not a wired publication implementation."""
from datetime import datetime, timedelta, timezone

import pytest

from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.io.channels import ChannelAddress
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.claim_fence import ClaimSuperseded, fenced_claim_transaction
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.questions import SQLiteQuestionStore


def test_existing_outbox_can_share_claim_transaction_and_roll_back_on_expiry(tmp_path):
    path = tmp_path / 'runtime.sqlite'
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    outbox = SQLiteOutboundStore(path)
    now = datetime.now(timezone.utc)
    queue.enqueue(CoreMessage(timestamp=now, prompt='synthetic', user='alice'), message_id='message')
    receipt = queue.claim_next().claim_receipt
    address = ChannelAddress('desktop', 'desktop', 'synthetic', alphonse_user_id='alice')

    # This proves only SQLite staging rollback; it doesn't establish runtime
    # routing, task/actor authority, dispatch safety, or hold integration.
    times = iter((now, now + timedelta(days=1)))
    with queue._connect() as conn, pytest.raises(ClaimSuperseded):
        with fenced_claim_transaction(conn, receipt, clock=lambda: next(times)):
            outbox.enqueue(address=address, message='synthetic', task_id='task', connection=conn)
    assert outbox.list_pending() == []

    with queue._connect() as conn:
        with fenced_claim_transaction(conn, receipt, clock=lambda: now):
            outbox.enqueue(address=address, message='synthetic', task_id='task', connection=conn)
    assert len(outbox.list_pending()) == 1
    # Fenced insertion alone is not fenced dispatch: current consumers can
    # claim the pending row even after its inbound claim has been reclaimed.
    assert queue.reclaim_expired(now=now + timedelta(days=1)) == 1
    assert outbox.claim_next().message == 'synthetic'


def test_outbox_rejects_connection_to_different_database(tmp_path):
    queue_path, other_path = tmp_path / 'queue.sqlite', tmp_path / 'other.sqlite'
    queue = SQLiteMessageQueue(queue_path, lease_owner='worker', experimental_claim_fencing=True)
    same_database_outbox = SQLiteOutboundStore(queue_path)
    other_outbox = SQLiteOutboundStore(other_path)
    now = datetime.now(timezone.utc)
    queue.enqueue(CoreMessage(timestamp=now, prompt='synthetic', user='alice'), message_id='message')
    receipt = queue.claim_next().claim_receipt
    address = ChannelAddress('desktop', 'desktop', 'synthetic', alphonse_user_id='alice')
    with queue._connect() as conn, pytest.raises(ValueError, match='outbox_connection_database_mismatch'):
        with fenced_claim_transaction(conn, receipt, clock=lambda: now):
            other_outbox.enqueue(address=address, message='synthetic', task_id='task', connection=conn)
    assert same_database_outbox.list_pending() == []
    assert other_outbox.list_pending() == []


def test_outbox_rejects_other_memory_database_and_accepts_its_own():
    first, second = SQLiteOutboundStore(), SQLiteOutboundStore()
    address = ChannelAddress('desktop', 'desktop', 'synthetic', alphonse_user_id='alice')
    with first._connect() as conn:
        with pytest.raises(ValueError, match='outbox_connection_database_mismatch'):
            second.enqueue(address=address, message='synthetic', connection=conn)
        first.enqueue(address=address, message='synthetic', connection=conn)
    assert len(first.list_pending()) == 1
    assert second.list_pending() == []


def test_outbox_connection_targets_main_even_with_temp_shadow(tmp_path):
    outbox = SQLiteOutboundStore(tmp_path / 'runtime.sqlite')
    address = ChannelAddress('desktop', 'desktop', 'synthetic', alphonse_user_id='alice')
    with outbox._connect() as conn:
        conn.execute('CREATE TEMP TABLE v2_outbox(outbox_message_id TEXT)')
        outbox.enqueue(address=address, message='synthetic', connection=conn, idempotency_key='synthetic')
        assert conn.execute('SELECT count(*) FROM temp.v2_outbox').fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM main.v2_outbox').fetchone()[0] == 1


def test_outbox_accepts_symlink_alias_for_same_database(tmp_path):
    import sqlite3

    path = tmp_path / 'runtime.sqlite'
    outbox = SQLiteOutboundStore(path)
    alias = tmp_path / 'alias.sqlite'
    alias.symlink_to(path)
    address = ChannelAddress('desktop', 'desktop', 'synthetic', alphonse_user_id='alice')
    with sqlite3.connect(alias) as conn:
        first = outbox.enqueue(address=address, message='synthetic', connection=conn, idempotency_key='alias')
        again = outbox.enqueue(address=address, message='synthetic', connection=conn, idempotency_key='alias')
        assert first.outbox_message_id == again.outbox_message_id
        assert conn.row_factory is None
    assert len(outbox.list_pending()) == 1


def test_live_claim_alone_does_not_fence_a_superseded_checkpoint(tmp_path):
    """A future publication boundary must bind task version, not just lease."""
    path = tmp_path / 'runtime.sqlite'
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    outbox = SQLiteOutboundStore(path)
    questions = SQLiteQuestionStore(path)
    now = datetime.now(timezone.utc)
    queue.enqueue(CoreMessage(timestamp=now, prompt='synthetic', user='alice'), message_id='message')
    receipt = queue.claim_next().claim_receipt
    task = TaskState(task_id='task', message_id='message', user='alice', goal='synthetic')
    original_version = questions.save_task_checkpoint(task, status='running')
    # A newer checkpoint may cancel or supersede work before the lease changes.
    task.goal = 'new synthetic instruction'
    assert questions.save_task_checkpoint(task, status='running') > original_version
    address = ChannelAddress('desktop', 'desktop', 'synthetic', alphonse_user_id='alice')
    with queue._connect() as conn:
        with fenced_claim_transaction(conn, receipt, clock=lambda: now):
            outbox.enqueue(address=address, message='stale synthetic response', task_id='task', connection=conn)
    # This is evidence of the missing check, not an acceptable host behavior.
    assert outbox.claim_next().message == 'stale synthetic response'


def test_expired_outbox_worker_can_settle_after_another_worker_reclaims(tmp_path):
    """The current outbox completion API carries no distinct claim receipt."""
    outbox = SQLiteOutboundStore(tmp_path / 'runtime.sqlite')
    address = ChannelAddress('desktop', 'desktop', 'synthetic', alphonse_user_id='alice')
    item = outbox.enqueue(address=address, message='synthetic')
    first = outbox.claim_next(lease_owner='first', lease_seconds=1)
    assert first.outbox_message_id == item.outbox_message_id
    assert outbox.reclaim_expired(now=datetime.now(timezone.utc) + timedelta(minutes=1)) == 1
    second = outbox.claim_next(lease_owner='second')
    assert second.outbox_message_id == item.outbox_message_id
    # The old caller can overwrite the new claim: a delivery receipt must bind
    # a unique acquisition, not merely the message id or reusable worker name.
    assert outbox.mark_delivered(first.outbox_message_id, provider_message_id='stale-receipt')
    assert outbox.get(item.outbox_message_id).status == 'delivered'
