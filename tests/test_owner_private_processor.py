"""Fake-provider qualification of the disabled-by-default owner knowledge host."""
import json
import os
import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore, _event
from alphonse.agent_v2.core.core import CoreLoopContext, ProcessingStatus
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_processor import OwnerReadConversationProcessor, authorize_trusted_owner_claim
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import SQLiteTaskBudgetStore
from alphonse.agent_v2.core.intelligence.simple_loop.owner_private_knowledge_read import build_private_knowledge_tool_definition
from alphonse.agent_v2.core.memory.contracts import AccessScope, conversation_item
from alphonse.agent_v2.core.memory.knowledge import KnowledgeRecord, KnowledgeStore, _source_snapshot_digest
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_read_memory_source_tool_definition, build_search_memory_tool_definition
from alphonse.agent_v2.users import V2UserStore


@pytest.mark.parametrize('mode', ['enabled', 'disabled', 'changed_in_flight', 'lost_response'])
def test_private_knowledge_opt_in_and_provider_freshness(tmp_path, mode):
    path = tmp_path / 'runtime.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    conversations = SQLiteConversationStore(path)
    KnowledgeStore(path)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    source = conversations.record(owner_user_id=owner.user_id, role='user', content='Synthetic apples.',
                                   source='test', source_message_id='old')
    item = conversation_item(source, owner.user_id)
    record = KnowledgeRecord(memory_id='memory:fixture', content='Synthetic apples.', access=AccessScope(owner.user_id),
        sources=(item.source,), recorded_at=source.created_at, kind='fact', subjects=('apples',), claim_key='apple',
        basis='user_report', quotes=('Synthetic apples.',), source_binding_version=1,
        source_snapshot_digests=(_source_snapshot_digest(item),))
    with queue._connect() as conn:
        conn.execute('INSERT INTO memory_knowledge VALUES (?,?,?,?,?,?)',
                     (record.memory_id, owner.user_id, record.claim_key, json.dumps(record.to_dict()), record.status, '[]'))
    message_id = queue.enqueue_trusted_local_owner(prompt='What do you remember about apples?', peer_uid=os.getuid())
    acquired = queue.claim_next(lease_seconds=60)
    task = TaskState.from_message(acquired.message, message_id=message_id)
    task.task_id = message_id
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    registry.register(build_private_knowledge_tool_definition())
    events, telemetry, handoff = [], [], []
    class Provider:
        supports_image_input = False
        def __init__(self): self.requests = []
        def generate_json(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                decision = {'kind': 'tool', 'tool_id': 'native.owner_knowledge', 'arguments': {}, 'text': None}
            else:
                if mode == 'lost_response':
                    raise TimeoutError('synthetic provider response lost')
                if mode == 'changed_in_flight':
                    with queue._connect() as conn:
                        conn.execute('UPDATE v2_conversation_events SET content=? WHERE event_id=?',
                                     ('Synthetic apples. More context.', source.event_id))
                        changed = conversation_item(_event(conn.execute('SELECT * FROM v2_conversation_events WHERE event_id=?',
                                                                      (source.event_id,)).fetchone()), owner.user_id)
                        from dataclasses import replace
                        rebound = replace(record, source_snapshot_digests=(_source_snapshot_digest(changed),))
                        conn.execute('UPDATE memory_knowledge SET payload=? WHERE memory_id=?',
                                     (json.dumps(rebound.to_dict()), record.memory_id))
                decision = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic apples.'}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 100, 'output_tokens': 20})
    provider = Provider()
    ctx = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt, tools=registry,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        conversation_store=conversations, cancellation_checker=lambda: False, ui_event_sink=events.append,
        telemetry_sink=telemetry.append)
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(path)),
        authorize_claim=authorize_trusted_owner_claim, save_checkpoint=lambda *_: None,
        atomic_checkpoints=ExperimentalCheckpointStore(SQLiteQuestionStore(path)),
        selected_draft_sink=lambda checkpoint, _ctx: handoff.append(checkpoint),
        enable_private_knowledge=mode != 'disabled')
    result = processor.process(task, ctx)
    if mode == 'enabled':
        assert result.status == ProcessingStatus.WAITING, result.error
        assert len(handoff) == 1
        assert 'Synthetic apples.' in provider.requests[1].prompt
        assert record.source_snapshot_digests[0] not in ''.join(r.prompt for r in provider.requests)
        assert 'Synthetic apples.' not in str((events, telemetry))
        # This constructor gate controls test rollout, not revocation of a
        # draft already selected under a live claim.
        processor.enable_private_knowledge = False
        picked = handoff[0].take_desktop_draft(ctx)
        assert picked['message'] == 'Synthetic apples.'
    elif mode in ('changed_in_flight', 'lost_response'):
        assert result.status == ProcessingStatus.FAILED
        assert not handoff
        assert len(provider.requests) == 2
    else:
        assert all('Synthetic apples.' not in r.prompt for r in provider.requests)
