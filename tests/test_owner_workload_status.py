"""The workload widget must not count retained requests as active work."""
from datetime import datetime, timezone
import os
from types import SimpleNamespace

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.messages.queue import MessageSelector
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.services.killswitch import KillSwitchCoordinator
from alphonse.agent_v2.users import V2UserStore


def test_owner_workload_separates_retained_processing_and_legacy_from_live_work(tmp_path):
    path = tmp_path / 'workload.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    SQLiteConversationStore(path)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    old = queue.enqueue_trusted_local_owner(prompt='Earlier request', peer_uid=os.getuid())
    queue.claim_next(selector=MessageSelector(message_id=old), lease_owner='old')
    queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt='Legacy request',
        user=owner.user_id), message_id='legacy')
    waiting = queue.enqueue_trusted_local_owner(prompt='New request', peer_uid=os.getuid())
    killswitch = KillSwitchCoordinator()
    daemon = SimpleNamespace(experimental_owner_only=True, runtime=SimpleNamespace(queue=queue),
                             kill_switch=killswitch, _owner_telegram_listener=None)

    assert V2Daemon._inbound_queue_status(daemon) == {'ready': 1, 'processing': 0, 'retry_wait': 0}
    queue.claim_next(selector=MessageSelector(message_id=waiting), lease_owner='worker')
    killswitch.activate(message_id=waiting, task_id=waiting, user_id=owner.user_id, project_id='', metadata={})
    assert V2Daemon._inbound_queue_status(daemon) == {'ready': 0, 'processing': 1, 'retry_wait': 0}
    killswitch.clear(waiting)
    assert V2Daemon._inbound_queue_status(daemon) == {'ready': 0, 'processing': 0, 'retry_wait': 0}
