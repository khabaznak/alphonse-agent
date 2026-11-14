"""Offline router-backed candidate decisions with disposable tools."""
import json

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.inference.models import InferencePurpose, InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateHost, CandidateState, run_candidate
from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits, SyntheticEvaluationBudgetStore, TaskBudget
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from alphonse.agent_v2.evaluation.router_candidate_chooser import RouterCandidateChooser


class ScriptedProvider:
    supports_image_input = False
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
    def generate_json(self, request):
        self.requests.append(request)
        value = next(self.responses)
        return InferenceResult(json_value=value, raw_response=json.dumps(value), usage={})


def make_router(provider):
    return InferenceRouter(provider=provider, default_profile=ModelProfile('fake', 'fixture', 'offline'))


def vector(actions=0, calls=0, inp=0, out=0):
    return {'actions': actions, 'model_calls': calls, 'input_tokens': inp, 'output_tokens': out}


def test_router_drives_candidate_read_and_answer_with_host_authorization(tmp_path):
    descriptor = ToolDescriptor('native.read_note', 'read_note', ToolKind.NATIVE, read_only=True,
        description='Read a note.', argument_schema={'type': 'object', 'properties': {'path': {'type': 'string'}},
                                                     'required': ['path']})
    provider = ScriptedProvider([
        {'kind': 'tool', 'tool_id': 'native.read_note', 'arguments': {'path': 'note'}, 'text': None},
        {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'forty two'},
    ])
    chooser = RouterCandidateChooser(router=make_router(provider), tools=(descriptor,), max_input_tokens=8192)
    registry = InMemoryToolRegistry()
    calls = []
    registry.register(ToolDefinition(descriptor, lambda args: calls.append(args) or
        {'candidate_complete': True, 'value': 'forty two'}))
    task = TaskState(task_id='task', user='owner')
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry)
    budgets = SyntheticEvaluationBudgetStore(str(tmp_path / 'budget.db'))
    budgets.create('task', TaskBudget(BudgetLimits(vector(2, 4, 40000, 4000), vector(0, 1, 20, 20),
                       100000, 1000), started_at_ms=0))
    evidence = {'goal': 'Read the note.'}
    def record(_actor, result):
        evidence['read'] = result['result']['value']
        return 'read'
    host = CandidateHost(task, context, budgets, lambda _: None, chooser,
        lambda actor, ref: actor == 'owner' and ref in evidence, lambda actor, ref: evidence[ref],
        lambda actor, tool, args: actor == 'owner' and tool == 'native.read_note' and args == {'path': 'note'},
        record, lambda actor, text, refs: text == 'forty two' and refs == ('read',), lambda: 100,
        input_reservation=8192, output_reservation=512, synthetic_allow_unreported_usage=True)
    result = run_candidate(CandidateState('task', 'owner', 'goal'), host)
    assert result.status == 'completed' and calls == [{'path': 'note'}]
    assert all(request.purpose is InferencePurpose.SIMPLE_LOOP_ACTION for request in provider.requests)
    assert 'native.read_note' in provider.requests[0].prompt
    assert 'forty two' in provider.requests[1].prompt
    assert len(budgets.load('task').usage()['unknown_usage']) == 2


def test_rejects_effectful_descriptors_and_does_not_trust_model_usage():
    router = make_router(ScriptedProvider([]))
    with pytest.raises(ValueError, match='tools_invalid'):
        RouterCandidateChooser(router=router, tools=(ToolDescriptor('native.send', 'send', ToolKind.NATIVE),),
                               max_input_tokens=4096)
    provider = ScriptedProvider([{'kind': 'answer', 'tool_id': None, 'arguments': None,
                                  'text': 'ok', 'usage': {'input_tokens': 1, 'output_tokens': 1}}])
    chooser = RouterCandidateChooser(router=make_router(provider), tools=(), max_input_tokens=4096)
    assert chooser('goal', 512)['usage'] == {'input_tokens': None, 'output_tokens': None}


def test_invalid_unlisted_tool_and_cancelled_call_never_become_tool_actions():
    provider = ScriptedProvider([{'kind': 'tool', 'tool_id': 'native.unknown', 'arguments': {}, 'text': None}])
    chooser = RouterCandidateChooser(router=make_router(provider), tools=(), max_input_tokens=4096)
    assert chooser('goal', 512)['kind'] == 'invalid'
    chooser.cancellation_checker = lambda: True
    with pytest.raises(ValueError, match='cancelled'):
        chooser('goal', 512)
    assert len(provider.requests) == 1
