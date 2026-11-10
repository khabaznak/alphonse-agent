"""Synthetic idempotency keys cannot silently select a different operation."""
from dataclasses import replace
import sqlite3

import pytest

from alphonse.agent_v2.core.io.channels import ChannelAddress
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore


@pytest.mark.parametrize('transaction', [False, True])
@pytest.mark.parametrize('change', ['audience', 'destination', 'message', 'task', 'kind', 'metadata'])
def test_idempotency_rejects_different_operation_without_mutation(tmp_path, transaction, change):
    store = SQLiteOutboundStore(tmp_path / 'outbox.sqlite')
    address = ChannelAddress('desktop', 'desktop', 'alice-private', alphonse_user_id='alice')
    original = dict(address=address, message='fixture secret', kind='response',
                    audience_user_id='alice', task_id='task-a', idempotency_key='fixed-fixture-key')
    first = store.enqueue(**original)
    altered = dict(original)
    if change == 'audience':
        altered['audience_user_id'] = 'bob'
    elif change == 'destination':
        altered['address'] = replace(address, channel_target='bob-private')
    elif change == 'message':
        altered['message'] = 'different fixture'
    elif change == 'task':
        altered['task_id'] = 'task-b'
    elif change == 'kind':
        altered['kind'] = 'other'
    else:
        altered['metadata'] = {'asset_ids': ['different-attachment']}
    with pytest.raises(ValueError, match='^outbox_idempotency_conflict$'):
        if transaction:
            with store._connect() as conn:
                store.enqueue(**altered, connection=conn)
        else:
            store.enqueue(**altered)
    assert store.get(first.outbox_message_id) == first
    assert len(store.list()) == 1


def test_repeated_identical_operation_returns_existing_delivery_state(tmp_path):
    store = SQLiteOutboundStore(tmp_path / 'outbox.sqlite')
    address = ChannelAddress('desktop', 'desktop', 'alice-private', alphonse_user_id='alice')
    kwargs = dict(address=address, message='fixture', audience_user_id='alice', task_id='task',
                  idempotency_key='same-operation')
    first = store.enqueue(**kwargs)
    assert store.mark_delivered(first.outbox_message_id, provider_message_id='receipt')
    repeated = store.enqueue(**kwargs)
    assert repeated.status == 'delivered'
    assert repeated.provider_message_id == 'receipt'
    assert len(store.list()) == 1


def test_unrelated_constraint_failure_does_not_return_a_phantom_message(tmp_path):
    store = SQLiteOutboundStore(tmp_path / 'outbox.sqlite')
    address = ChannelAddress('desktop', 'desktop', 'alice-private', alphonse_user_id='alice')
    with store._connect() as conn:
        conn.execute('CREATE UNIQUE INDEX fixture_unique_kind ON v2_outbox(kind)')
    store.enqueue(address=address, message='first fixture', kind='response')
    with pytest.raises(sqlite3.IntegrityError):
        store.enqueue(address=address, message='second fixture', kind='response')
    assert len(store.list()) == 1


def test_in_memory_retry_ignores_temporary_table_shadow():
    store = SQLiteOutboundStore()
    address = ChannelAddress('desktop', 'desktop', 'alice-private', alphonse_user_id='alice')
    first = store.enqueue(address=address, message='fixture', idempotency_key='same')
    with store._connect() as conn:
        conn.execute('CREATE TEMP TABLE v2_outbox(outbox_message_id TEXT)')
    repeated = store.enqueue(address=address, message='fixture', idempotency_key='same')
    assert repeated.outbox_message_id == first.outbox_message_id


def test_in_memory_consumers_ignore_temporary_table_shadow():
    store = SQLiteOutboundStore()
    address = ChannelAddress('desktop', 'desktop', 'alice-private', alphonse_user_id='alice')
    first = store.enqueue(address=address, message='trusted fixture', idempotency_key='one')
    with store._connect() as conn:
        conn.execute('CREATE TEMP TABLE v2_outbox AS SELECT * FROM main.v2_outbox')
        conn.execute("UPDATE temp.v2_outbox SET message='forged fixture'")
    assert store.get(first.outbox_message_id).message == 'trusted fixture'
    assert store.claim_next().message == 'trusted fixture'
    assert store.mark_delivered(first.outbox_message_id, provider_message_id='receipt')
    with store._connect() as conn:
        assert conn.execute('SELECT status FROM main.v2_outbox').fetchone()[0] == 'delivered'
        assert conn.execute('SELECT status FROM temp.v2_outbox').fetchone()[0] == 'pending'


def test_shared_connection_reclaim_ignores_temporary_table_shadow(tmp_path):
    from datetime import datetime, timedelta, timezone
    store = SQLiteOutboundStore(tmp_path / 'outbox.sqlite')
    address = ChannelAddress('desktop', 'desktop', 'alice-private', alphonse_user_id='alice')
    item = store.enqueue(address=address, message='fixture')
    claimed = store.claim_next(lease_seconds=1)
    assert claimed.outbox_message_id == item.outbox_message_id
    with store._connect() as conn:
        conn.execute('CREATE TEMP TABLE v2_outbox AS SELECT * FROM main.v2_outbox')
        assert store.reclaim_expired(conn=conn, now=datetime.now(timezone.utc) + timedelta(minutes=2)) == 1
        assert conn.execute('SELECT status FROM main.v2_outbox').fetchone()[0] == 'pending'
        assert conn.execute('SELECT status FROM temp.v2_outbox').fetchone()[0] == 'claimed'


def test_reclaim_rejects_connection_from_other_outbox(tmp_path):
    from datetime import datetime, timedelta, timezone
    first, second = SQLiteOutboundStore(tmp_path / 'one.sqlite'), SQLiteOutboundStore(tmp_path / 'two.sqlite')
    address = ChannelAddress('desktop', 'desktop', 'alice-private', alphonse_user_id='alice')
    item = second.enqueue(address=address, message='fixture')
    assert second.claim_next(lease_seconds=1).outbox_message_id == item.outbox_message_id
    with second._connect() as conn:
        with pytest.raises(ValueError, match='^outbox_connection_database_mismatch$'):
            first.reclaim_expired(conn=conn, now=datetime.now(timezone.utc) + timedelta(minutes=2))
    assert second.get(item.outbox_message_id).status == 'claimed'


def test_reclaim_requires_own_persistent_in_memory_connection():
    first, second = SQLiteOutboundStore(), SQLiteOutboundStore()
    with second._connect() as conn:
        with pytest.raises(ValueError, match='^outbox_connection_database_mismatch$'):
            first.reclaim_expired(conn=conn)
    with first._connect() as conn:
        assert first.reclaim_expired(conn=conn) == 0


def test_regenerated_ack_conflict_keeps_one_message_and_continues_task(tmp_path):
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.inference import InferenceResult
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
    from alphonse.agent_v2.core.io import build_outbox_delivery_sink, channel_metadata
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.system_one import SystemOneAdmissionDecision

    class Inference:
        def __init__(self, text):
            self.text = text
        def generate_json(self, request):
            return InferenceResult(json_value={'requested_outcome': 'fixture request', 'supplied_parameters': [],
                'unresolved_dependencies': [], 'constraints': [], 'superseded_constraints': []})
        def generate_markdown(self, request):
            return InferenceResult(content=self.text, model_profile=request.model_profile)

    class Judge:
        def classify_task_admission(self, **kwargs):
            return SystemOneAdmissionDecision(True, 0.97, True)
        def review_task_acknowledgement(self, **kwargs):
            return True

    def task():
        return TaskState(task_id='same-task', message_id='same-task', goal='fixture request', user='alice',
            metadata={'routing_disposition': 'pdca_task', 'channel': channel_metadata(
                integration_id='desktop', provider_key='tui', channel_target='alice', alphonse_user_id='alice')})

    store = SQLiteOutboundStore(tmp_path / 'outbox.sqlite')
    events = []
    def context(message):
        return CoreLoopContext(messages=InMemoryMessageQueue(), inference=Inference(message), system_one=Judge(),
            delivery_sink=build_outbox_delivery_sink(outbox=store), activity_sink=events.append)

    original, resumed = task(), task()
    assert HierarchicalCAPDProcessor._admit_initial_human_task(original, context('I will check.')) is False
    events.clear()
    assert HierarchicalCAPDProcessor._admit_initial_human_task(resumed, context('I will look into it.')) is False
    assert resumed.metadata['v3_admission']['acknowledgement'] == {'status': 'delivery_failed', 'error_type': 'ValueError'}
    assert resumed.status != 'completed'
    assert not any(event.label == 'request acknowledged' for event in events)
    assert [item.message for item in store.list()] == ['I will check.']
