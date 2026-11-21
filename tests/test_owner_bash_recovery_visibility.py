"""Owner sees an unresolved Bash effect without disclosing commands or replaying it."""
from datetime import datetime, timezone
import json
import os
from types import SimpleNamespace

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.simple_loop.owner_atomic_checkpoint import OwnerAtomicCheckpoint
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import BudgetLimits, TaskBudget
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon, _owner_bash_action_unconfirmed
from alphonse.agent_v2.users import V2UserStore


def test_expired_owner_bash_reservation_is_visible_without_replay_or_command(tmp_path):
    path = tmp_path / 'owner.db'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    SQLiteOutboundStore(path)
    SQLiteConversationStore(path)
    store = ExperimentalCheckpointStore(SQLiteQuestionStore(path))
    message_id = queue.enqueue_trusted_local_owner(prompt='Synthetic request', peer_uid=os.getuid())
    acquired = queue.claim_next()
    task = TaskState.from_message(acquired.message, message_id=message_id)
    task.task_id = message_id
    atomic = OwnerAtomicCheckpoint(queue, store, task, acquired.claim_receipt, clock=lambda: datetime.now(timezone.utc))
    vector = lambda a, m, i, o: {'actions': a, 'model_calls': m, 'input_tokens': i, 'output_tokens': o}
    start = atomic.initialize(CandidateState(message_id, owner.user_id, 'goal'),
        TaskBudget(BudgetLimits(vector(3, 3, 1000, 300), vector(0, 1, 100, 50), 100000, 0), started_at_ms=100))
    decision = {'kind': 'tool', 'tool_id': 'native.bash', 'arguments': {'command': 'synthetic-private-command'}}
    model = atomic.reserve(expected_version=start.version, reservation_id='model', kind='model',
        amounts=vector(0, 1, 100, 50), now_ms=100)
    chosen = atomic.settle_model(expected_version=model.version, reservation_id='model', outcome='success',
        input_tokens=1, output_tokens=1, decision=decision, projection_fingerprint='a' * 64)
    atomic.reserve(expected_version=chosen.version, reservation_id='shell', kind='tool', amounts=vector(1, 0, 0, 0),
        now_ms=101, tool_id='native.bash', arguments=decision['arguments'], selected_decision=decision,
        projection_fingerprint='a' * 64)
    with queue._connect() as conn:
        conn.execute("UPDATE v2_inbound_messages SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE message_id=?", (message_id,))
        original = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (message_id,)).fetchone()[0]
    daemon = object.__new__(V2Daemon)
    daemon.runtime = SimpleNamespace(queue=queue)
    daemon.owner_message_status = lambda _: 'unavailable'
    items = daemon.owner_recovery_list()
    assert items == [{'message_id': message_id, 'prompt': 'Synthetic request',
                      'status': 'action_unconfirmed', 'can_abandon': False}]
    assert 'synthetic-private-command' not in repr(items)
    with queue._connect() as conn:
        assert conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (message_id,)).fetchone()[0] == original
    raw = json.loads(original)
    assert not _owner_bash_action_unconfirmed(raw, message_id=message_id, actor='another-user')
    bad_schema = json.loads(original)
    from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_checkpoint import STATE_KEY
    bad_schema['metadata'][STATE_KEY]['schema_version'] = True
    assert not _owner_bash_action_unconfirmed(bad_schema, message_id=message_id, actor=owner.user_id)
    raw['metadata']['owner_read_pending_action']['reservation_id'] = 'other'
    assert not _owner_bash_action_unconfirmed(raw, message_id=message_id, actor=owner.user_id)
