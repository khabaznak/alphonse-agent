"""Owner search defaults cover common result sets without accepting partial pages."""
import json

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext, ProcessingStatus
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_processor import OwnerReadConversationProcessor
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import SQLiteTaskBudgetStore
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import (
    build_read_memory_source_tool_definition, build_search_memory_tool_definition,
)
from alphonse.agent_v2.users import V2UserStore


@pytest.mark.parametrize('count,expected', [(9, ProcessingStatus.WAITING), (31, ProcessingStatus.FAILED), (91, ProcessingStatus.FAILED)])
def test_owner_search_default_is_bounded_and_never_accepts_partial_page(tmp_path, count, expected):
    users = V2UserStore(':memory:')
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    store = SQLiteConversationStore(tmp_path / 'conversation.db')
    for i in range(count):
        store.record(owner_user_id=owner.user_id, role='user', content=f'Apples note {i}' + (' unique' if i == 30 else ''),
                     source='test', source_message_id=str(i))
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())

    class Provider:
        supports_image_input = False
        def __init__(self):
            self.requests = []
        def generate_json(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                action = {'kind': 'tool', 'tool_id': 'native.search_memory',
                          'arguments': {'query': 'apples', 'source_types': ['conversation']}, 'text': None}
            else:
                action = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic answer.'}
            return InferenceResult(json_value=action, raw_response=json.dumps(action),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    provider = Provider()
    task = TaskState(task_id='message', message_id='message', user=owner.user_id, goal='Find apples.',
                     intelligence_engine='single_loop_experimental', intelligence_schema_version=1)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        conversation_store=store, cancellation_checker=lambda: False)
    checkpoints = []
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'budget.db')),
        authorize_claim=lambda *_: True, save_checkpoint=lambda _task, snapshot: checkpoints.append(snapshot),
        now_ms=lambda: 100)
    result = processor.process(task, context)
    assert result.status == expected
    if count == 9:
        assert len(provider.requests) == 2
        assert all(f'Apples note {i}' in provider.requests[1].prompt for i in range(count))
    else:
        assert len(provider.requests) == 2
        assert 'next_offset' in provider.requests[1].prompt
        if count > 90:
            assert 'too_many_matches' in provider.requests[1].prompt
        assert 'Apples note' not in provider.requests[1].prompt


def test_complete_search_projection_labels_speakers_without_duplicate_or_private_content(tmp_path):
    users = V2UserStore(':memory:')
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    store = SQLiteConversationStore(tmp_path / 'conversation.db')
    for owner_id, role, content, identifier in (
        (owner.user_id, 'user', 'Plan is amber lantern.', '1'),
        (owner.user_id, 'assistant', 'Plan is unchosen lantern.', '2'),
        ('other-owner', 'user', 'Private lantern information.', '3'),
    ):
        store.record(owner_user_id=owner_id, role=role, content=content, source='test', source_message_id=identifier)
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())

    class Provider:
        supports_image_input = False
        def __init__(self):
            self.requests = []
        def generate_json(self, request):
            self.requests.append(request)
            decision = ({'kind': 'tool', 'tool_id': 'native.search_memory',
                         'arguments': {'query': 'lantern', 'source_types': ['conversation']}, 'text': None}
                        if len(self.requests) == 1 else
                        {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic answer.'})
            return InferenceResult(json_value=decision, usage={'input_tokens': 100, 'output_tokens': 20})

    provider = Provider()
    task = TaskState(task_id='message', message_id='message', user=owner.user_id, goal='Find lantern.',
                     intelligence_engine='single_loop_experimental', intelligence_schema_version=1)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        conversation_store=store, cancellation_checker=lambda: False)
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'budget.db')),
        authorize_claim=lambda *_: True, save_checkpoint=lambda *_: None, now_ms=lambda: 100)
    assert processor.process(task, context).status == ProcessingStatus.WAITING
    prompt = provider.requests[1].prompt
    assert '"speaker": "user"' in prompt and '"speaker": "assistant"' in prompt
    assert prompt.count('Plan is amber lantern.') == 1
    assert prompt.count('Plan is unchosen lantern.') == 1
    assert 'Private lantern information.' not in prompt
    assert 'ordered by search relevance, not chronology' in prompt


@pytest.mark.parametrize('reverse', [False, True])
def test_owner_can_narrow_an_overflowing_search_without_seeing_partial_results(tmp_path, reverse):
    users = V2UserStore(':memory:')
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    store = SQLiteConversationStore(tmp_path / 'conversation.db')
    for i in range(31):
        store.record(owner_user_id=owner.user_id, role='user', content=f'Apples note {i}' + (' unique' if i == 30 else ''),
                     source='test', source_message_id=str(i))
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())

    class Provider:
        supports_image_input = False
        def __init__(self):
            self.requests = []
        def generate_json(self, request):
            self.requests.append(request)
            if len(self.requests) <= 2:
                query = ('unique' if reverse else 'apples') if len(self.requests) == 1 else ('apples' if reverse else 'unique')
                action = {'kind': 'tool', 'tool_id': 'native.search_memory',
                          'arguments': {'query': query, 'source_types': ['conversation']}, 'text': None}
            else:
                action = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic answer.'}
            return InferenceResult(json_value=action, raw_response=json.dumps(action),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    provider = Provider()
    task = TaskState(task_id='message', message_id='message', user=owner.user_id, goal='Find apples.',
                     intelligence_engine='single_loop_experimental', intelligence_schema_version=1)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        conversation_store=store, cancellation_checker=lambda: False)
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'budget.db')),
        authorize_claim=lambda *_: True, save_checkpoint=lambda *_: None, now_ms=lambda: 100)
    result = processor.process(task, context)
    assert len(provider.requests) == 3
    if reverse:
        # A complete older search cannot clear the later overflow.
        assert result.status == ProcessingStatus.FAILED
        assert result.error == 'candidate_answer_unsupported'
        assert 'next_offset' in provider.requests[2].prompt
    else:
        assert result.status == ProcessingStatus.WAITING, result.error
        assert 'next_offset' in provider.requests[1].prompt
        assert 'Apples note' not in provider.requests[1].prompt
        assert 'conversation:' not in provider.requests[1].prompt
        assert 'Apples note 30' in provider.requests[2].prompt

@pytest.mark.parametrize('count', [31, 61])
def test_owner_search_collects_complete_snapshot_before_exposing_results(tmp_path, count):
    users = V2UserStore(':memory:')
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    store = SQLiteConversationStore(tmp_path / 'conversation.db')
    for i in range(count):
        store.record(owner_user_id=owner.user_id, role='user', content=f'Apples record {i}',
                     source='test', source_message_id=str(i))
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())

    class Provider:
        supports_image_input = False
        def __init__(self):
            self.requests = []
        def generate_json(self, request):
            self.requests.append(request)
            page = len(self.requests) - 1
            if page < (count + 29) // 30:
                arguments = {'query': 'apples', 'source_types': ['conversation']}
                if page:
                    arguments['offset'] = 30 * page
                    assert 'Apples record' not in request.prompt
                action = {'kind': 'tool', 'tool_id': 'native.search_memory', 'arguments': arguments, 'text': None}
            else:
                assert all(f'Apples record {i}' in request.prompt for i in range(count))
                action = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic answer.'}
            return InferenceResult(json_value=action, raw_response=json.dumps(action),
                                   usage={'input_tokens': 100, 'output_tokens': 20})
    provider = Provider()
    task = TaskState(task_id='message', message_id='message', user=owner.user_id, goal='Find apples.',
                     intelligence_engine='single_loop_experimental', intelligence_schema_version=1)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        conversation_store=store, cancellation_checker=lambda: False)
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'budget.db')),
        authorize_claim=lambda *_: True, save_checkpoint=lambda *_: None, now_ms=lambda: 100)
    result = processor.process(task, context)
    assert result.status == ProcessingStatus.WAITING, result.error


def test_owner_search_rejects_changed_snapshot_before_next_page(tmp_path):
    users = V2UserStore(':memory:')
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    store = SQLiteConversationStore(tmp_path / 'conversation.db')
    for i in range(31):
        store.record(owner_user_id=owner.user_id, role='user', content=f'Apples record {i}',
                     source='test', source_message_id=str(i))
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    class Provider:
        supports_image_input = False
        def __init__(self): self.requests = []
        def generate_json(self, request):
            self.requests.append(request)
            args = {'query': 'apples', 'source_types': ['conversation']}
            if len(self.requests) > 1:
                assert 'Apples record' not in request.prompt
                store.record(owner_user_id=owner.user_id, role='user', content='Another apples record',
                             source='test', source_message_id='new')
                args['offset'] = 30
            action = {'kind': 'tool', 'tool_id': 'native.search_memory', 'arguments': args, 'text': None}
            return InferenceResult(json_value=action, raw_response=json.dumps(action),
                                   usage={'input_tokens': 100, 'output_tokens': 20})
    provider = Provider()
    task = TaskState(task_id='message', message_id='message', user=owner.user_id, goal='Find apples.',
                     intelligence_engine='single_loop_experimental', intelligence_schema_version=1)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        conversation_store=store, cancellation_checker=lambda: False)
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'budget.db')),
        authorize_claim=lambda *_: True, save_checkpoint=lambda *_: None, now_ms=lambda: 100)
    assert processor.process(task, context).status == ProcessingStatus.FAILED
    assert len(provider.requests) == 2


def test_owner_search_at_page_budget_asks_to_narrow(tmp_path):
    users = V2UserStore(':memory:')
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    store = SQLiteConversationStore(tmp_path / 'conversation.db')
    for i in range(4):
        store.record(owner_user_id=owner.user_id, role='user', content=f'Apples record {i}',
                     source='test', source_message_id=str(i))
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    class Provider:
        supports_image_input = False
        def __init__(self): self.requests = []
        def generate_json(self, request):
            self.requests.append(request)
            index = len(self.requests) - 1
            if index == 3:
                assert 'too_many_matches' in request.prompt
                assert 'Apples record' not in request.prompt
                action = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic answer.'}
            else:
                args = {'query': 'apples', 'source_types': ['conversation'], 'limit': 1}
                if index: args['offset'] = index
                action = {'kind': 'tool', 'tool_id': 'native.search_memory', 'arguments': args, 'text': None}
            return InferenceResult(json_value=action, raw_response=json.dumps(action),
                                   usage={'input_tokens': 100, 'output_tokens': 20})
    provider = Provider()
    task = TaskState(task_id='message', message_id='message', user=owner.user_id, goal='Find apples.',
                     intelligence_engine='single_loop_experimental', intelligence_schema_version=1)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        conversation_store=store, cancellation_checker=lambda: False)
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'budget.db')),
        authorize_claim=lambda *_: True, save_checkpoint=lambda *_: None, now_ms=lambda: 100)
    assert processor.process(task, context).status == ProcessingStatus.FAILED
    assert len(provider.requests) == 4
