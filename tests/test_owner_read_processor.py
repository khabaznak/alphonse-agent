"""Disposable proof of the real owner-read host; no daemon registration."""
import json
import os
from dataclasses import fields, replace
from datetime import datetime, timezone
from types import SimpleNamespace

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import AlphonseCore, CoreLoopContext, CoreMessage, LoopStepStatus, ProcessingStatus, StateSnapshot
from alphonse.agent_v2.core.inference.models import InferencePurpose, InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_processor import OwnerReadConversationProcessor, authorize_trusted_owner_claim
from alphonse.agent_v2.core.intelligence.engine_router import EngineRoutingProcessor
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import SQLiteTaskBudgetStore
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.io.outbox import project_snapshot_to_outbox
import pytest
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.state import reset_state
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import (
    build_read_memory_source_tool_definition, build_search_memory_tool_definition,
)
from alphonse.agent_v2.users import V2UserStore


class Provider:
    supports_image_input = False
    def __init__(self, source_id):
        self.requests = []
        self.source_id = source_id
    def generate_json(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            result = {'kind': 'tool', 'tool_id': 'native.search_memory',
                      'arguments': {'query': 'apples', 'source_types': ['conversation']}, 'text': None}
        elif len(self.requests) == 2:
            result = {'kind': 'tool', 'tool_id': 'native.read_memory_source',
                      'arguments': {'source_id': self.source_id, 'neighbors': 0}, 'text': None}
        else:
            result = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic apples.'}
        return InferenceResult(json_value=result, raw_response=json.dumps(result),
                               usage={'input_tokens': 100, 'output_tokens': 20})


def test_owner_candidate_uses_real_memory_and_blocks_replay(tmp_path):
    users = V2UserStore(':memory:')
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    member = users.create_user(display_name='Synthetic member')
    conversation = SQLiteConversationStore(tmp_path / 'conversation.db')
    event = conversation.record(owner_user_id=owner.user_id, role='user', content='Synthetic apples.', source='test', source_message_id='one')
    conversation.record(owner_user_id=member.user_id, role='user', content='Synthetic apples private.', source='test', source_message_id='two')
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    provider = Provider('conversation:' + event.event_id)
    router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline'))
    task = TaskState(task_id='message-1', message_id='message-1', user=owner.user_id,
        goal='Find the apples note.', intelligence_engine='single_loop_experimental', intelligence_schema_version=1)
    ctx = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, inference=router,
                          conversation_store=conversation, cancellation_checker=lambda: False)
    saved = []
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'budget.db')),
        authorize_claim=lambda task, context: True, save_checkpoint=lambda task, snapshot: saved.append(snapshot),
        now_ms=lambda: 100)
    result = processor.process(task, ctx)
    assert result.status == ProcessingStatus.WAITING, result.error
    assert result.snapshot.metadata['draft'] == 'Synthetic apples.'
    assert result.snapshot.metadata['delivery'] == 'not_integrated'
    with pytest.raises(ValueError, match='candidate_response_publication_unavailable'):
        project_snapshot_to_outbox(snapshot=result.snapshot, outbox=None)
    assert len(provider.requests) == 3 and 'Synthetic apples.' in provider.requests[1].prompt
    assert all(request.purpose is InferencePurpose.SIMPLE_LOOP_ACTION
               and request.user == owner.user_id and request.task_id == task.task_id
               for request in provider.requests)
    assert 'Synthetic apples.' in provider.requests[2].prompt
    assert 'Search the owner’s prior conversation messages' in provider.requests[0].prompt
    assert 'Synthetic apples private.' not in provider.requests[1].prompt
    assert saved[-1]['status'] == 'completed'
    replay = processor.process(task, ctx)
    assert replay.status == ProcessingStatus.FAILED and replay.error == 'owner_read_initialization_unavailable'
    assert len(provider.requests) == 3


@pytest.mark.parametrize('read_source', [True, False])
@pytest.mark.parametrize('include_unrelated', [False, True])
def test_owner_search_truncated_excerpt_is_discovery_only(tmp_path, read_source, include_unrelated):
    users = V2UserStore(':memory:')
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    conversation = SQLiteConversationStore(tmp_path / 'conversation.db')
    event = conversation.record(owner_user_id=owner.user_id, role='user',
        content='Synthetic willow picnic Thursday. ' + ('Routine planning. ' * 100) + 'Moved to Saturday.',
        source='test', source_message_id='long-fixture')
    unrelated = None
    if include_unrelated:
        unrelated = conversation.record(owner_user_id=owner.user_id, role='user',
            content='Willow picnic seating inventory: plates and chairs.',
            source='test', source_message_id='other-fixture')
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())

    class TruncatedProvider:
        supports_image_input = False
        def __init__(self):
            self.requests = []
        def generate_json(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                decision = {'kind': 'tool', 'tool_id': 'native.search_memory',
                            'arguments': {'query': 'willow picnic', 'source_types': ['conversation']}, 'text': None}
            elif len(self.requests) == 2 and read_source:
                assert 'Shortened previews are withheld' in request.prompt
                assert 'Thursday' not in request.prompt
                assert 'Moved to Saturday.' not in request.prompt
                assert event.event_id in request.prompt
                if unrelated:
                    assert unrelated.event_id in request.prompt
                    assert 'seating inventory' in request.prompt
                decision = {'kind': 'tool', 'tool_id': 'native.read_memory_source',
                            'arguments': {'source_id': 'conversation:' + event.event_id, 'neighbors': 0}, 'text': None}
            else:
                if read_source:
                    assert 'Moved to Saturday.' in request.prompt
                decision = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Saturday.'}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    provider = TruncatedProvider()
    router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline'))
    task = TaskState(task_id='message-long', message_id='message-long', user=owner.user_id,
                     goal='What is the current willow picnic day?',
                     intelligence_engine='single_loop_experimental', intelligence_schema_version=1)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, inference=router,
                              conversation_store=conversation, cancellation_checker=lambda: False)
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'budget.db')),
        authorize_claim=lambda *_: True, save_checkpoint=lambda *_: None, now_ms=lambda: 100)
    result = processor.process(task, context)
    if read_source:
        assert result.status == ProcessingStatus.WAITING, result.error
        assert result.snapshot.metadata['draft'] == 'Saturday.'
        # One relevant full source is enough; the unrelated search hit is not
        # forced to be read. This does not prove semantic grounding.
        assert len(provider.requests) == 3
    elif include_unrelated:
        # The other matching source is complete. This allows a model to use
        # complete search evidence without reading an unrelated long source;
        # it does not prove this scripted answer is grounded.
        assert result.status == ProcessingStatus.WAITING
        assert len(provider.requests) == 2
    else:
        assert result.status == ProcessingStatus.FAILED
        assert len(provider.requests) == 2


def test_shortened_preview_is_hidden_even_if_model_guesses_after_reading_other_source(tmp_path):
    users = V2UserStore(':memory:')
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    conversation = SQLiteConversationStore(tmp_path / 'conversation.db')
    conversation.record(owner_user_id=owner.user_id, role='user',
        content='Willow picnic Thursday. ' + ('Planning details. ' * 100) + 'Update: Saturday.',
        source='test', source_message_id='fixture-a')
    other = conversation.record(owner_user_id=owner.user_id, role='user',
        content='Willow picnic equipment: blanket and plates.', source='test', source_message_id='fixture-b')
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())

    class WrongSourceProvider:
        supports_image_input = False
        def __init__(self):
            self.requests = []
        def generate_json(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                decision = {'kind': 'tool', 'tool_id': 'native.search_memory',
                            'arguments': {'query': 'willow picnic', 'source_types': ['conversation']}, 'text': None}
            elif len(self.requests) == 2:
                assert 'Thursday' not in request.prompt
                assert 'Willow picnic equipment: blanket and plates.' in request.prompt
                decision = {'kind': 'tool', 'tool_id': 'native.read_memory_source',
                            'arguments': {'source_id': 'conversation:' + other.event_id, 'neighbors': 0}, 'text': None}
            else:
                # The model can still guess from its own output even though
                # the shortened source's content was withheld.
                assert 'Thursday' not in request.prompt
                decision = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Thursday.'}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    provider = WrongSourceProvider()
    router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline'))
    task = TaskState(task_id='wrong-source', message_id='wrong-source', user=owner.user_id,
                     goal='When is the willow picnic?', intelligence_engine='single_loop_experimental',
                     intelligence_schema_version=1)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, inference=router,
                              conversation_store=conversation, cancellation_checker=lambda: False)
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'budget.db')),
        authorize_claim=lambda *_: True, save_checkpoint=lambda *_: None, now_ms=lambda: 100)
    result = processor.process(task, context)
    # Diagnostic of the remaining semantic limitation, not an oracle pass:
    # the complete unrelated hit allows an answer, which is wrong here.
    assert result.status == ProcessingStatus.WAITING
    assert result.snapshot.metadata['draft'] == 'Thursday.'
    assert result.snapshot.metadata['draft'] != 'Saturday.'
    assert len(provider.requests) == 3


def test_complete_search_result_does_not_force_unrelated_overlong_source_read(tmp_path):
    users = V2UserStore(':memory:')
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    conversation = SQLiteConversationStore(tmp_path / 'conversation.db')
    conversation.record(owner_user_id=owner.user_id, role='user',
        content='Willow picnic: the latest plan is Saturday.', source='test', source_message_id='relevant')
    conversation.record(owner_user_id=owner.user_id, role='user',
        content='Willow picnic equipment list. ' + ('unrelated supplies. ' * 800),
        source='test', source_message_id='long-unrelated')
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())

    class Provider:
        supports_image_input = False
        def __init__(self): self.requests = []
        def generate_json(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                decision = {'kind': 'tool', 'tool_id': 'native.search_memory',
                            'arguments': {'query': 'willow picnic', 'source_types': ['conversation']}, 'text': None}
            else:
                assert 'the latest plan is Saturday' in request.prompt
                assert 'unrelated supplies' not in request.prompt
                decision = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Saturday.'}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    provider = Provider()
    task = TaskState(task_id='short-source', message_id='short-source', user=owner.user_id,
        goal='What is the latest willow picnic day?', intelligence_engine='single_loop_experimental',
        intelligence_schema_version=1)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        conversation_store=conversation, cancellation_checker=lambda: False)
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'budget.db')),
        authorize_claim=lambda *_: True, save_checkpoint=lambda *_: None, now_ms=lambda: 100)
    result = processor.process(task, context)
    assert result.status == ProcessingStatus.WAITING, result.error
    assert result.snapshot.metadata['draft'] == 'Saturday.'  # independent synthetic oracle
    assert len(provider.requests) == 2  # no unrelated source read


def test_owner_host_requires_trusted_claim_and_active_owner(tmp_path):
    users = V2UserStore(':memory:')
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    task = TaskState(task_id='message-1', message_id='message-1', user=owner.user_id,
        goal='Hello', intelligence_engine='single_loop_experimental', intelligence_schema_version=1)
    ctx = CoreLoopContext(messages=InMemoryMessageQueue())
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'budget.db')),
        authorize_claim=lambda task, context: False, save_checkpoint=lambda *_: None)
    assert processor.process(task, ctx).error == 'owner_read_authority_unavailable'
    users.update_user(owner.user_id, is_active=False)
    processor.authorize_claim = lambda *_: True
    assert processor.process(task, ctx).error == 'owner_read_authority_unavailable'


@pytest.mark.parametrize('outcome', ['failed', 'cancelled'])
def test_owner_non_draft_outcomes_cannot_enter_legacy_outbox(tmp_path, monkeypatch, outcome):
    users = V2UserStore(':memory:')
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    task = TaskState(task_id='message-1', message_id='message-1', user=owner.user_id,
        goal='Hello', intelligence_engine='single_loop_experimental', intelligence_schema_version=1)
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    ctx = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry,
        inference=InferenceRouter(provider=Provider('unused'), default_profile=ModelProfile('test', 'fixture', 'offline')),
        conversation_store=SQLiteConversationStore(tmp_path / 'conversation.db'), cancellation_checker=lambda: False)
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'budget.db')),
        authorize_claim=lambda *_: outcome != 'failed', save_checkpoint=lambda *_: None)
    if outcome == 'cancelled':
        monkeypatch.setattr('alphonse.agent_v2.core.intelligence.simple_loop.owner_read_processor.run_candidate',
                            lambda *_: SimpleNamespace(status='cancelled'))
    result = processor.process(task, ctx)
    assert result.status == (ProcessingStatus.FAILED if outcome == 'failed' else ProcessingStatus.CANCELLED)
    with pytest.raises(ValueError, match='candidate_response_publication_unavailable'):
        project_snapshot_to_outbox(snapshot=result.snapshot, outbox=None)


def test_real_trusted_claim_is_carried_internally_and_verified(tmp_path):
    database = tmp_path / 'claim.db'
    users = V2UserStore(database)
    owner = users.onboard(display_name='Fixture owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(database, experimental_claim_fencing=True)
    from alphonse.agent_v2.conversations import SQLiteConversationStore
    SQLiteConversationStore(database)
    message_id = queue.enqueue_trusted_local_owner(prompt='Fixture', peer_uid=os.getuid())
    acquired = queue.claim_next()
    task = TaskState.from_message(acquired.message, message_id=message_id)
    task.task_id = message_id
    context = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt)
    assert 'claim_receipt' not in {field.name for field in fields(context)}
    assert 'claim_receipt' not in repr(context)
    assert authorize_trusted_owner_claim(task, context)
    assert not authorize_trusted_owner_claim(replace(task, user='other'), context)
    context.claim_receipt = None
    assert not authorize_trusted_owner_claim(task, context)
    context.claim_receipt = acquired.claim_receipt
    users.update_user(owner.user_id, is_active=False)
    assert not authorize_trusted_owner_claim(task, context)


def test_injected_owner_candidate_runs_from_queue_with_real_memory(tmp_path):
    reset_state()
    users = V2UserStore(':memory:')
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    conversation = SQLiteConversationStore(tmp_path / 'conversation.db')
    event = conversation.record(owner_user_id=owner.user_id, role='user', content='Synthetic apples.',
                                source='test', source_message_id='one')
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    provider = Provider('conversation:' + event.event_id)
    inference = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline'))
    saved = []
    claims = []
    def authorize(task, context):
        claims.append((task.task_id, task.message_id, task.user))
        return task.task_id == task.message_id == 'queue-message' and task.user == owner.user_id
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'budget.db')),
        authorize_claim=authorize, save_checkpoint=lambda task, snapshot: saved.append(snapshot), now_ms=lambda: 100)
    queue = InMemoryMessageQueue()
    state = SimpleNamespace(value=None)
    state.update = lambda value: setattr(state, 'value', value)
    state.snapshot = lambda: state.value or StateSnapshot()
    core = AlphonseCore(intelligence=EngineRoutingProcessor(owner_read_candidate=processor),
        messages=queue, tools=registry, prompts=object(), state=state, memory=None,
        conversation_store=conversation, inference=inference,
        question_store=SQLiteQuestionStore(tmp_path / 'questions.db'))
    queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt='Find the apples note.', user=owner.user_id,
        metadata={'intelligence_engine': 'single_loop_experimental', 'intelligence_schema_version': 1}),
        message_id='queue-message')
    result = core.step()
    assert result.status is LoopStepStatus.WAITING, result.error
    assert claims and all(claim == ('queue-message', 'queue-message', owner.user_id) for claim in claims)
    assert len(provider.requests) == 3 and 'Synthetic apples.' in provider.requests[2].prompt
    assert saved[-1]['status'] == 'completed'
    assert state.value.metadata['draft'] == 'Synthetic apples.'
    assert state.value.metadata['delivery'] == 'not_integrated'
    reset_state()
    replay_queue = InMemoryMessageQueue()
    core.messages = replay_queue
    replay_queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt='Find the apples note.', user=owner.user_id,
        metadata={'intelligence_engine': 'single_loop_experimental', 'intelligence_schema_version': 1}),
        message_id='queue-message')
    replay = core.step()
    assert replay.status is LoopStepStatus.FAILED
    assert replay.error == 'owner_read_initialization_unavailable'
    assert len(provider.requests) == 3
    reset_state()
