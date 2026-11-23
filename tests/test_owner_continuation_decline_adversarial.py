"""Owner refusal persists across reconstruction and stale UI choices."""
import pytest

from test_owner_atomic_checkpoint import setup
from test_owner_continuation_question import paused_checkpoint
from test_owner_continuation_desktop import daemon_for
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_reclaim import reclaim_approved_pause
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_checkpoint import CheckpointConflict


def test_decline_after_restart_cannot_be_replaced_by_stale_approval(setup):
    queue, acquired, task, store, _, _ = paused_checkpoint(setup)
    before = daemon_for(queue, store)
    item = before.owner_continuation_list()[0]
    assert before.owner_continuation_choose(task.message_id, item['checkpoint_version'], False) == {'status':'declined'}
    after = daemon_for(queue, store)
    assert after.owner_continuation_list() == []
    with pytest.raises(CheckpointConflict):
        after.owner_continuation_choose(task.message_id, item['checkpoint_version'], True)
    with pytest.raises(CheckpointConflict):
        reclaim_approved_pause(queue, store, task.message_id, lease_owner='worker', cancellation_checker=lambda: False)
    with queue._connect() as conn:
        row = conn.execute('SELECT claim_generation FROM v2_inbound_messages WHERE message_id=?', (task.message_id,)).fetchone()
    assert row['claim_generation'] == acquired.claim_receipt.claim_generation
