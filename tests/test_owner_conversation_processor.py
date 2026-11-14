"""Disposable core integration; no household provider, store, or delivery."""
from datetime import datetime, timezone
import json
from types import SimpleNamespace

from alphonse.agent_v2.core.core import (AlphonseCore, CoreLoopContext, CoreMessage,
    LoopStepStatus, ProcessingStatus, StateSnapshot, ToolDescriptor, ToolKind)
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.inference.models import InferencePurpose, InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.v3.owner_conversation_processor import SyntheticOwnerConversationProcessor
from alphonse.agent_v2.core.intelligence.v3.processor import EngineRoutingProcessor
from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateHost, CandidateState
from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits, SyntheticEvaluationBudgetStore, TaskBudget
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.state import reset_state
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from alphonse.agent_v2.evaluation.router_candidate_chooser import RouterCandidateChooser


def _vector(actions=0, model_calls=0, input_tokens=0, output_tokens=0):
    return dict(actions=actions, model_calls=model_calls, input_tokens=input_tokens, output_tokens=output_tokens)


def _host_factory(tmp_path, calls, *, read=False, chooser=None):
    def factory(task, context):
        store = SyntheticEvaluationBudgetStore(str(tmp_path / 'synthetic.db'))
        store.create(task.task_id, TaskBudget(BudgetLimits(_vector(1, 3, 30000, 2000),
            _vector(0, 1, 100, 50), 100000, 1000), started_at_ms=0))
        sources = {'goal': task.goal}
        def choose(projection, _cap):
            calls.append(projection)
            if read and len(calls) == 1:
                return {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {},
                        'usage': {'input_tokens': 10, 'output_tokens': 10}}
            return {'kind': 'answer', 'text': 'Synthetic answer',
                    'usage': {'input_tokens': 10, 'output_tokens': 10}}
        def record(actor, result):
            sources['observation'] = result['result']['value']
            return 'observation'
        host = CandidateHost(task, context, store, lambda _: None, chooser or choose,
            lambda actor, ref: actor == 'owner' and ref in sources,
            lambda actor, ref: sources[ref], lambda actor, tool, args: read and actor == 'owner' and tool == 'native.fixture_read', record,
            lambda actor, text, refs: actor == 'owner' and text == 'Synthetic answer' and (not read or 'observation' in refs), lambda: 100,
            input_reservation=8192, output_reservation=512)
        return host, CandidateState(task.task_id, task.user, 'goal')
    return factory


def test_queued_fixture_message_reaches_candidate_but_is_not_marked_delivered(tmp_path, monkeypatch):
    reset_state()
    from alphonse.agent_v2.core.intelligence.v3 import owner_conversation_processor as module
    original = module.run_candidate
    outcomes = []
    def capture(*args):
        outcome = original(*args)
        outcomes.append(outcome)
        return outcome
    monkeypatch.setattr(module, 'run_candidate', capture)
    queue = InMemoryMessageQueue()
    calls = []
    reads = []
    registry = InMemoryToolRegistry()
    descriptor = ToolDescriptor('native.fixture_read', 'fixture_read', ToolKind.NATIVE, read_only=True,
                                description='Read the disposable synthetic source.')
    registry.register(ToolDefinition(descriptor,
        lambda args: reads.append(args) or {'candidate_complete': True, 'value': 'Synthetic source value'}))
    class Provider:
        def __init__(self):
            self.requests = []
            self.responses = iter([
                {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {}, 'text': None},
                {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic answer'},
            ])
        def generate_json(self, request):
            self.requests.append(request)
            value = next(self.responses)
            return InferenceResult(json_value=value, raw_response=json.dumps(value),
                                   usage={'input_tokens': 10, 'output_tokens': 10})
    provider = Provider()
    router = InferenceRouter(provider=provider, default_profile=ModelProfile('fake', 'fixture', 'offline'))
    chooser = RouterCandidateChooser(router=router, tools=(descriptor,), max_input_tokens=8192)
    processor = SyntheticOwnerConversationProcessor(owner='owner', host_factory=_host_factory(tmp_path, calls, read=True, chooser=chooser))
    state = SimpleNamespace(value=None)
    state.update = lambda value: setattr(state, 'value', value)
    state.snapshot = lambda: state.value or StateSnapshot()
    core = AlphonseCore(intelligence=EngineRoutingProcessor(v3=None, synthetic_owner_candidate=processor),
        messages=queue, tools=registry, prompts=object(), state=state, memory=None)
    queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt='Synthetic question', user='owner',
        metadata={'intelligence_engine': 'synthetic_owner_candidate', 'intelligence_schema_version': 1}),
        message_id='fixture-message')
    result = core.step()
    assert result.status is LoopStepStatus.WAITING, [(x.status, x.blocker) for x in outcomes]
    assert len(provider.requests) == 2 and all(r.purpose is InferencePurpose.SIMPLE_LOOP_ACTION for r in provider.requests)
    assert 'Synthetic question' in provider.requests[0].prompt
    assert reads == [{}] and 'Synthetic source value' in provider.requests[1].prompt
    assert state.value.metadata['draft'] == 'Synthetic answer'
    assert state.value.metadata['delivery'] == 'not_integrated'
    assert state.value.metadata['status'] == 'draft_only'
    reset_state()


def test_no_default_registration_or_nonowner_execution(tmp_path):
    task = TaskState(task_id='task', message_id='message', user='other',
        intelligence_engine='synthetic_owner_candidate', intelligence_schema_version=1)
    context = CoreLoopContext(messages=InMemoryMessageQueue())
    calls = []
    processor = SyntheticOwnerConversationProcessor(owner='owner', host_factory=_host_factory(tmp_path, calls))
    result = processor.process(task, context)
    assert result.status is ProcessingStatus.FAILED
    assert result.error == 'synthetic_owner_context_unavailable' and not calls
    import pytest
    with pytest.raises(ValueError, match='intelligence_engine_unregistered'):
        EngineRoutingProcessor(v3=None).process(task, context)


def test_same_message_is_not_reexecuted_by_same_fixture_processor(tmp_path):
    task = TaskState(task_id='task', message_id='message', user='owner', goal='Synthetic question')
    context = CoreLoopContext(messages=InMemoryMessageQueue())
    calls = []
    processor = SyntheticOwnerConversationProcessor(owner='owner', host_factory=_host_factory(tmp_path, calls))
    assert processor.process(task, context).status is ProcessingStatus.WAITING
    result = processor.process(task, context)
    assert result.status is ProcessingStatus.FAILED
    assert result.error == 'synthetic_owner_replay_unavailable' and len(calls) == 1


def test_fixture_refuses_a_real_memory_or_publication_sink_before_host_creation(tmp_path):
    task = TaskState(task_id='task', message_id='message', user='owner', goal='Synthetic question')
    calls = []
    processor = SyntheticOwnerConversationProcessor(owner='owner', host_factory=_host_factory(tmp_path, calls))
    for context in (CoreLoopContext(messages=InMemoryMessageQueue(), memory=object()),
                    CoreLoopContext(messages=InMemoryMessageQueue(), ui_event_sink=lambda _: None)):
        result = processor.process(task, context)
        assert result.status is ProcessingStatus.FAILED
        assert result.error == 'synthetic_owner_context_unavailable'
    assert not calls
