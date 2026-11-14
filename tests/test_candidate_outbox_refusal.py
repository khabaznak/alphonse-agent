"""Candidate responses must not enter the legacy unfenced delivery path."""

from types import SimpleNamespace
from datetime import datetime, timezone

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import StateSnapshot
from alphonse.agent_v2.core.io.channels import channel_metadata
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.database import transaction


@pytest.mark.parametrize('location', ['snapshot', 'task', 'task_metadata'])
def test_candidate_snapshot_refused_before_outbox_or_transcript_write(tmp_path, location):
    path = tmp_path / 'household.sqlite'
    outbox = SQLiteOutboundStore(path)
    conversations = SQLiteConversationStore(path)
    daemon = V2Daemon.__new__(V2Daemon)
    daemon.runtime = SimpleNamespace(
        outbox=outbox, conversation_store=conversations, identity_resolver=None,
        user_store=SimpleNamespace(mirror_automation_messages_to_preferred_channel=lambda: False),
    )
    task = {
        'task_id': 'task-1', 'user': 'owner', 'status': 'completed',
        'intelligence_engine': 'hierarchical_v3',
        'metadata': {'channel': channel_metadata(integration_id='desktop', provider_key='desktop',
                                                   channel_target='owner', alphonse_user_id='owner'),
                     'prepared_user_response': {'message': 'Synthetic answer'}},
    }
    snapshot = StateSnapshot(metadata={'task_state': task})
    if location == 'snapshot':
        snapshot.metadata['engine'] = 'single_loop_experimental'
    elif location == 'task':
        task['intelligence_engine'] = 'single_loop_experimental'
    else:
        task['metadata']['intelligence_engine'] = 'single_loop_experimental'

    with pytest.raises(ValueError, match='candidate_response_publication_unavailable'):
        with transaction(path, immediate=True) as connection:
            daemon._project_snapshot_delivery(snapshot, connection=connection)
    assert outbox.list() == []
    assert conversations.list_owner(owner_user_id='owner') == []


def test_checkpoint_failure_stops_retry_and_preserves_saved_task(tmp_path):
    from alphonse.agent_v2.core.core import CoreMessage, LoopStepStatus
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.messages import SQLiteMessageQueue
    from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
    from alphonse.agent_v2.runtime import build_runtime_host
    from alphonse.agent_v2.users import V2UserStore

    users = V2UserStore(':memory:')
    owner = users.onboard(display_name='Owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(':memory:')
    runtime = build_runtime_host(user_store=users, messages=queue,
                                 schedule_store=ScheduledTaskStore(':memory:'))
    runtime.core.memory = None
    daemon = V2Daemon(runtime)
    message = queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt='Resume work',
        user=owner.user_id, metadata={'intelligence_engine': 'hierarchical_v3',
            'intelligence_schema_version': 3, 'channel': channel_metadata(
                integration_id='desktop', provider_key='desktop', channel_target=owner.user_id,
                alphonse_user_id=owner.user_id)}))
    saved = TaskState(task_id=message.message_id, message_id=message.message_id,
                      user=owner.user_id, goal='Earlier work', facts_md='Preserve evidence',
                      intelligence_engine='single_loop_experimental', intelligence_schema_version=1)
    runtime.question_store.save_task_checkpoint(saved, status='running')

    class RejectingProcessor:
        def process(self, task, context):
            raise ValueError('candidate_checkpoint_unavailable')
    runtime.core.intelligence = RejectingProcessor()
    step = daemon.run_once()

    assert step.status is LoopStepStatus.FAILED
    assert queue.status_for(message.message_id) == 'failed'
    restored = runtime.question_store.load_task_checkpoint(message.message_id)
    assert restored is not None and restored.facts_md == 'Preserve evidence'
    assert restored.intelligence_engine == 'single_loop_experimental'
    notices = runtime.outbox.list()
    assert len(notices) == 1 and 'previous state needs review' in notices[0].message
    assert 'candidate_checkpoint_unavailable' not in notices[0].message
