"""Desktop acknowledgement must not overwrite a newer outbox claim."""
from datetime import datetime, timedelta, timezone

import pytest

from alphonse.agent_v2.core.io.channels import ChannelAddress
from alphonse.agent_v2.core.io.outbox import OutboundSelector, SQLiteOutboundStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host


@pytest.mark.parametrize('next_owner', ['desktop:other', 'desktop:original'])
def test_reclaim_between_desktop_read_and_ack_preserves_new_claim(tmp_path, monkeypatch, next_owner):
    outbox = SQLiteOutboundStore(tmp_path / 'outbox.sqlite')
    runtime = build_runtime_host(outbox=outbox)
    daemon = V2Daemon(runtime)
    item = outbox.enqueue(address=ChannelAddress('desktop', 'desktop', 'alice', alphonse_user_id='alice'),
                          message='synthetic')
    first = outbox.claim_next(OutboundSelector(integration_id='desktop'), lease_owner='desktop:original')
    assert first is not None
    real_ack = outbox.acknowledge_desktop_claim
    callbacks = []
    monkeypatch.setattr(daemon, '_on_outbox_delivered', callbacks.append)

    def reclaim_after_read(delivery, *, lease_owner):
        assert outbox.reclaim_expired(now=datetime.now(timezone.utc) + timedelta(days=1)) == 1
        assert outbox.claim_next(OutboundSelector(integration_id='desktop'), lease_owner=next_owner) is not None
        return real_ack(delivery, lease_owner=lease_owner)

    monkeypatch.setattr(outbox, 'acknowledge_desktop_claim', reclaim_after_read)
    assert not daemon.acknowledge_desktop_delivery(client_id='original', outbox_message_id=item.outbox_message_id)
    current = outbox.get(item.outbox_message_id)
    assert (current.status, current.lease_owner, current.attempt_count) == ('claimed', next_owner, 2)
    assert callbacks == []


def test_expired_desktop_claim_is_not_acknowledged_before_reclaim(tmp_path, monkeypatch):
    outbox = SQLiteOutboundStore(tmp_path / 'outbox.sqlite')
    daemon = V2Daemon(build_runtime_host(outbox=outbox))
    item = outbox.enqueue(address=ChannelAddress('desktop', 'desktop', 'alice', alphonse_user_id='alice'),
                          message='synthetic')
    outbox.claim_next(OutboundSelector(integration_id='desktop'), lease_owner='desktop:original')
    with outbox._connect() as conn:
        conn.execute("UPDATE v2_outbox SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE outbox_message_id=?",
                     (item.outbox_message_id,))
    monkeypatch.setattr(daemon, '_on_outbox_delivered', lambda *_: pytest.fail('expired claim published'))
    assert not daemon.acknowledge_desktop_delivery(client_id='original', outbox_message_id=item.outbox_message_id)
    assert outbox.get(item.outbox_message_id).status == 'claimed'


def test_desktop_ack_rolls_back_if_lease_expires_during_write(tmp_path, monkeypatch):
    from alphonse.agent_v2.core.io import outbox as outbox_module
    outbox = SQLiteOutboundStore(tmp_path / 'outbox.sqlite')
    daemon = V2Daemon(build_runtime_host(outbox=outbox))
    item = outbox.enqueue(address=ChannelAddress('desktop', 'desktop', 'alice', alphonse_user_id='alice'),
                          message='synthetic')
    claim = outbox.claim_next(OutboundSelector(integration_id='desktop'), lease_owner='desktop:original')
    expires = datetime.fromisoformat(claim.lease_expires_at)
    samples = iter((expires - timedelta(seconds=1), expires))
    monkeypatch.setattr(outbox_module, '_now', lambda: next(samples))
    monkeypatch.setattr(daemon, '_on_outbox_delivered', lambda *_: pytest.fail('expired claim published'))
    assert not daemon.acknowledge_desktop_delivery(client_id='original', outbox_message_id=item.outbox_message_id)
    assert outbox.get(item.outbox_message_id).status == 'claimed'
