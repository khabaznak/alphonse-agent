"""Only the current owner sees unresolved Telegram delivery states."""
from types import SimpleNamespace
import pytest

from test_owner_telegram_handoff import ready
from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_handoff import (
    stage_telegram_reply, dispatch_telegram_reply, finalize_telegram_reply,
)
from alphonse.agent_v2.daemon import V2Daemon


def recovery_daemon(queue):
    daemon = V2Daemon.__new__(V2Daemon)
    daemon.runtime = SimpleNamespace(queue=queue)
    return daemon


@pytest.mark.parametrize('accepted', [False, True])
def test_direct_telegram_delivery_needs_owner_scoped_private_recovery(ready, accepted):
    queue, atomic, context = ready
    stage_telegram_reply(atomic, context)
    status = dispatch_telegram_reply(atomic, context, send=lambda **_: 'provider-id' if accepted else '')
    assert status == ('provider_accepted' if accepted else 'started_unconfirmed')
    daemon = recovery_daemon(queue)
    matches = [item for item in daemon.owner_recovery_list() if item['message_id'] == atomic.task.message_id]
    assert len(matches) == 1
    assert matches[0]['status'] == ('telegram_reconciliation_required' if accepted else 'telegram_delivery_unconfirmed')
    assert matches[0]['can_abandon'] is False
    assert matches[0]['prompt'] == 'Telegram response'
    assert 'Hello from Alphonse' not in str(matches[0])
    if accepted:
        finalize_telegram_reply(atomic, context)
        assert not any(item['message_id'] == atomic.task.message_id for item in daemon.owner_recovery_list())
    else:
        with queue._connect() as conn:
            conn.execute("UPDATE v2_owner_telegram_replies SET owner_id='other-user' WHERE message_id=?", (atomic.task.message_id,))
        assert not any(item['message_id'] == atomic.task.message_id for item in daemon.owner_recovery_list())
