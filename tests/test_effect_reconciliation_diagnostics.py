"""Diagnostics: current legacy outbox does not reconcile ordinary sends.

These describe missing safety properties, not acceptable candidate behavior.
The disposable provider below has applied the effect before losing its reply.
"""
from alphonse.agent_v2.core.io.channels import ChannelAddress
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore


def test_ordinary_timeout_can_cause_automatic_duplicate_attempt(tmp_path):
    store = SQLiteOutboundStore(tmp_path / 'runtime.sqlite')
    address = ChannelAddress('fixture', 'telegram', 'synthetic-chat', alphonse_user_id='alice')
    record = store.enqueue(address=address, message='synthetic', task_id='task', audience_user_id='alice')
    effects = []
    first = store.claim_next(lease_owner='first')
    effects.append(first.message)  # Provider accepted, then its response was lost.
    assert store.mark_failed(first.outbox_message_id, error='synthetic timeout', retry_after_seconds=0)
    second = store.claim_next(lease_owner='second')
    assert second.outbox_message_id == record.outbox_message_id
    effects.append(second.message)
    assert effects == ['synthetic', 'synthetic']


def test_scheduled_timeout_retains_unknown_without_auto_retry(tmp_path):
    store = SQLiteOutboundStore(tmp_path / 'runtime.sqlite')
    store.scheduled_delivery_guard = lambda _: True
    address = ChannelAddress('fixture', 'telegram', 'synthetic-chat', alphonse_user_id='alice')
    record = store.enqueue(address=address, message='synthetic', task_id='task', audience_user_id='alice',
                           metadata={'scheduled_task_id': 'schedule', 'occurrence_key': 'occurrence'})
    first = store.claim_next(lease_owner='first')
    assert store.mark_failed(first.outbox_message_id, error='synthetic timeout', retry_after_seconds=0)
    assert store.get(record.outbox_message_id).status == 'unconfirmed'
    assert store.claim_next(lease_owner='second') is None
