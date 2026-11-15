"""Synthetic gym wiring and independent oracle; no live model or household data."""
import json


from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext, ProcessingStatus
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_processor import OwnerReadConversationProcessor
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import SQLiteTaskBudgetStore
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_read_memory_source_tool_definition, build_search_memory_tool_definition
from alphonse.agent_v2.evaluation.family_memory_gym import PROMPT, SOURCES, score
from alphonse.agent_v2.users import V2UserStore


def test_family_memory_case_uses_relevant_conversation_sources(tmp_path):
    users = V2UserStore(':memory:')
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    store = SQLiteConversationStore(tmp_path / 'conversation.db')
    ids = {key: 'conversation:' + store.record(owner_user_id=owner.user_id, role='user',
           content=value, source='synthetic-gym', source_message_id=key).event_id
           for key, value in SOURCES.items()}
    tools = InMemoryToolRegistry()
    tools.register(build_search_memory_tool_definition())
    tools.register(build_read_memory_source_tool_definition())

    class ScriptedChooser:
        supports_image_input = False
        def __init__(self):
            self.steps = []
        def generate_json(self, request):
            index = len(self.steps)
            if index == 0:
                action = {'kind': 'tool', 'tool_id': 'native.search_memory',
                          'arguments': {'query': 'family', 'source_types': ['conversation']}, 'text': None}
            elif index < 3:
                key = ('people', 'pet')[index - 1]
                assert ids[key] in request.prompt
                action = {'kind': 'tool', 'tool_id': 'native.read_memory_source',
                          'arguments': {'source_id': ids[key], 'neighbors': 0}, 'text': None}
            else:
                assert SOURCES['people'] in request.prompt and SOURCES['pet'] in request.prompt
                action = {'kind': 'answer', 'tool_id': None, 'arguments': None,
                          'text': 'I remember Mira and Theo live in the household, and their dog is Pepper.'}
            self.steps.append(action)
            return InferenceResult(json_value=action, raw_response=json.dumps(action),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    chooser = ScriptedChooser()
    task = TaskState(task_id='family-gym', message_id='family-gym', user=owner.user_id, goal=PROMPT,
                     intelligence_engine='single_loop_experimental', intelligence_schema_version=1)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=tools,
        inference=InferenceRouter(provider=chooser, default_profile=ModelProfile('test', 'fixture', 'offline')),
        conversation_store=store, cancellation_checker=lambda: False)
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'budget.db')),
        authorize_claim=lambda *_: True, save_checkpoint=lambda *_: None, now_ms=lambda: 100)
    result = processor.process(task, context)
    assert result.status == ProcessingStatus.WAITING, result.error
    read_ids = {a['arguments']['source_id'] for a in chooser.steps if a.get('tool_id') == 'native.read_memory_source'}
    read_keys = {key for key, value in ids.items() if value in read_ids}
    assert all(score(result.snapshot.metadata['draft'], read_keys).values())
    # This verifies the host and scoring path under scripted decisions only.


def test_family_memory_oracle_rejects_missing_sources_or_invented_claims():
    checks = score('Mira and Theo have children and a dog named Pepper.', {'people'})
    assert checks['mentions_people'] and checks['mentions_pet']
    assert not checks['reads_relevant_sources'] and not checks['does_not_invent_child']
    assert not score('Mira, Theo, dog Pepper, projector.', {'people', 'pet', 'unrelated'})['omits_unrelated_source']
