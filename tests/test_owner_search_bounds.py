"""Owner search defaults cover common result sets without accepting partial pages."""
import json

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext, ProcessingStatus
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.owner_read_processor import OwnerReadConversationProcessor
from alphonse.agent_v2.core.intelligence.v3.task_budget import SQLiteTaskBudgetStore
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import (
    build_read_memory_source_tool_definition, build_search_memory_tool_definition,
)
from alphonse.agent_v2.users import V2UserStore


@pytest.mark.parametrize('count,expected', [(9, ProcessingStatus.WAITING), (31, ProcessingStatus.FAILED)])
def test_owner_search_default_is_bounded_and_never_accepts_partial_page(tmp_path, count, expected):
    users = V2UserStore(':memory:')
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    store = SQLiteConversationStore(tmp_path / 'conversation.db')
    for i in range(count):
        store.record(owner_user_id=owner.user_id, role='user', content=f'Apples note {i}',
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
        assert len(provider.requests) == 1
