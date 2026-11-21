"""Offline router-backed candidate decisions with disposable tools."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.inference.models import InferencePurpose, InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateHost, CandidateState, run_candidate
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import BudgetLimits, SQLiteTaskBudgetStore, SyntheticEvaluationBudgetStore, TaskBudget
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from alphonse.agent_v2.core.intelligence.simple_loop.router_candidate_chooser import RouterCandidateChooser


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
    task = TaskState(task_id='task', user='owner')
    chooser = RouterCandidateChooser(router=make_router(provider), task=task, tools=(descriptor,), max_input_tokens=8192)
    registry = InMemoryToolRegistry()
    calls = []
    registry.register(ToolDefinition(descriptor, lambda args: calls.append(args) or
        {'candidate_complete': True, 'value': 'forty two'}))
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
    assert all((request.task_id, request.user, request.project_id) == ('task', 'owner', '')
               for request in provider.requests)
    assert 'native.read_note' in provider.requests[0].prompt
    assert 'forty two' in provider.requests[1].prompt
    assert len(budgets.load('task').usage()['unknown_usage']) == 2


def test_rejects_effectful_descriptors_and_does_not_trust_model_usage():
    router = make_router(ScriptedProvider([]))
    task = TaskState(task_id='task', user='owner')
    with pytest.raises(ValueError, match='tools_invalid'):
        RouterCandidateChooser(router=router, task=task, tools=(ToolDescriptor('native.send', 'send', ToolKind.NATIVE),),
                               max_input_tokens=4096)
    provider = ScriptedProvider([{'kind': 'answer', 'tool_id': None, 'arguments': None,
                                  'text': 'ok', 'usage': {'input_tokens': 1, 'output_tokens': 1}}])
    chooser = RouterCandidateChooser(router=make_router(provider), task=task, tools=(), max_input_tokens=4096)
    assert chooser('goal', 512)['usage'] == {'input_tokens': None, 'output_tokens': None}


def test_disposable_bash_is_opt_in_and_does_not_allow_other_mutations():
    from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
    descriptor = build_bash_tool_definition().descriptor_with_schema()
    router = make_router(ScriptedProvider([]))
    task = TaskState(task_id='task', user='owner')
    with pytest.raises(ValueError, match='tools_invalid'):
        RouterCandidateChooser(router=router, task=task, tools=(descriptor,), max_input_tokens=8192)
    chooser = RouterCandidateChooser(router=router, task=task, tools=(descriptor,),
                                    max_input_tokens=8192, allow_disposable_bash=True)
    assert chooser._tool_ids == frozenset({'native.bash'})
    with pytest.raises(ValueError, match='tools_invalid'):
        RouterCandidateChooser(router=router, task=task, tools=(ToolDescriptor('native.send', 'send', ToolKind.NATIVE),),
                               max_input_tokens=8192, allow_disposable_bash=True)


def test_trusted_host_bash_is_an_explicit_opt_in_only():
    from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
    descriptor = build_bash_tool_definition().descriptor_with_schema()
    router = make_router(ScriptedProvider([]))
    task = TaskState(task_id='task', user='owner')
    with pytest.raises(ValueError, match='tools_invalid'):
        RouterCandidateChooser(router=router, task=task, tools=(descriptor,), max_input_tokens=8192)
    chooser = RouterCandidateChooser(router=router, task=task, tools=(descriptor,), max_input_tokens=8192,
                                     allow_owner_host_bash=True)
    assert chooser._tool_ids == frozenset({'native.bash'})
    with pytest.raises(ValueError, match='tools_invalid'):
        RouterCandidateChooser(router=router, task=task,
            tools=(ToolDescriptor('native.send', 'send', ToolKind.NATIVE),), max_input_tokens=8192,
            allow_owner_host_bash=True)


def test_codex_usage_settles_durable_budget_for_multi_step_candidate(tmp_path, monkeypatch):
    from alphonse.agent_v2.core.inference.openai_codex import OpenAICodexProvider
    descriptor = ToolDescriptor('native.read_note', 'read_note', ToolKind.NATIVE, read_only=True,
        description='Read a note.', argument_schema={'type': 'object', 'properties': {}})
    decisions = iter([
        {'kind': 'tool', 'tool_id': 'native.read_note', 'arguments': '{}', 'text': None},
        {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'observed'},
    ])
    def run(command, **_kwargs):
        if command[-2:] == ['login', 'status']:
            return SimpleNamespace(returncode=0, stderr='', stdout='Logged in using ChatGPT')
        Path(command[command.index('--output-last-message') + 1]).write_text(json.dumps(next(decisions)))
        return SimpleNamespace(returncode=0, stderr='', stdout=json.dumps(
            {'type': 'turn.completed', 'usage': {'input_tokens': 200, 'output_tokens': 20}}))
    monkeypatch.setattr('alphonse.agent_v2.core.inference.openai_codex.shutil.which', lambda _: '/bin/codex')
    monkeypatch.setattr('alphonse.agent_v2.core.inference.openai_codex.subprocess.run', run)
    router = InferenceRouter(provider=OpenAICodexProvider(),
        default_profile=ModelProfile('openai_codex', 'fixture', 'offline'))
    task = TaskState(task_id='task', user='owner')
    chooser = RouterCandidateChooser(router=router, task=task, tools=(descriptor,), max_input_tokens=8192)
    registry = InMemoryToolRegistry()
    registry.register(ToolDefinition(descriptor, lambda _: {'candidate_complete': True, 'value': 'observed'}))
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry)
    budgets = SQLiteTaskBudgetStore(str(tmp_path / 'durable-budget.db'))
    budgets.create('task', TaskBudget(BudgetLimits(vector(1, 4, 40000, 4000), vector(0, 1, 20, 20),
                       100000, 1000), started_at_ms=0))
    evidence = {'goal': 'Read the note.'}
    def record(_actor, result):
        evidence['read'] = result['result']['value']
        return 'read'
    host = CandidateHost(task, context, budgets, lambda _: None, chooser,
        lambda actor, ref: actor == 'owner' and ref in evidence, lambda actor, ref: evidence[ref],
        lambda actor, tool, args: actor == 'owner' and tool == 'native.read_note' and args == {},
        record, lambda actor, text, refs: text == 'observed' and refs == ('read',), lambda: 100,
        input_reservation=8192, output_reservation=512)
    result = run_candidate(CandidateState('task', 'owner', 'goal'), host)
    assert result.status == 'completed'
    usage = budgets.load('task').usage()
    assert usage['unknown_usage'] == [] and usage['in_flight'] == []
    assert usage['charged']['input_tokens'] == 400
    assert usage['charged']['output_tokens'] == 40


def test_invalid_unlisted_tool_and_cancelled_call_never_become_tool_actions():
    provider = ScriptedProvider([{'kind': 'tool', 'tool_id': 'native.unknown', 'arguments': {}, 'text': None}])
    task = TaskState(task_id='task', user='owner')
    chooser = RouterCandidateChooser(router=make_router(provider), task=task, tools=(), max_input_tokens=4096)
    assert chooser('goal', 512)['kind'] == 'invalid'
    chooser.cancellation_checker = lambda: True
    with pytest.raises(ValueError, match='cancelled'):
        chooser('goal', 512)
    assert len(provider.requests) == 1


@pytest.mark.parametrize('when', ['before', 'during'])
def test_task_identity_change_is_rejected_before_using_decision(when):
    task = TaskState(task_id='task', user='owner', project_id='home')
    class Provider:
        calls = 0
        def generate_json(self, request):
            self.calls += 1
            assert (request.task_id, request.user, request.project_id) == ('task', 'owner', 'home')
            task.user = 'other'
            return InferenceResult(json_value={'kind': 'answer', 'tool_id': None,
                'arguments': None, 'text': 'answer'}, usage={'input_tokens': 1, 'output_tokens': 1})
    provider = Provider()
    chooser = RouterCandidateChooser(router=make_router(provider), task=task, tools=(), max_input_tokens=4096)
    if when == 'before':
        task.user = 'other'
    with pytest.raises(ValueError, match='candidate_router_task_identity_changed'):
        chooser('goal', 512)
    assert provider.calls == (0 if when == 'before' else 1)


def test_gym_and_runtime_share_decision_policy_and_reject_unknown_tool():
    from alphonse.agent_v2.evaluation.codex_candidate_chooser import CodexSubscriptionChooser
    from alphonse.agent_v2.evaluation.codex_subscription_provider import CodexSubscriptionProvider

    invalid = {'kind': 'tool', 'tool_id': 'native.unknown', 'arguments': {}, 'text': None}
    model = ScriptedProvider([invalid])
    runtime = RouterCandidateChooser(router=make_router(model), task=TaskState(task_id='t', user='owner'),
        tools=(), max_input_tokens=8192)
    assert runtime('goal', 512)['kind'] == 'invalid'

    captured = []
    provider = CodexSubscriptionProvider(model='gpt-6-luna')
    def generate(request):
        captured.append(request)
        return InferenceResult(json_value=invalid, raw_response=json.dumps(invalid))
    provider.generate_json = generate
    gym = CodexSubscriptionChooser(provider=provider, model='gpt-6-luna',
        max_input_tokens=8192, emit=lambda _: None)
    assert gym('goal', 512)['kind'] == 'invalid'
    # Only the tool catalogs differ: selection/uncertainty guidance is shared.
    marker = 'Before answering, consider whether another available authorized source'
    assert model.requests[0].prompt.split(marker, 1)[1] == captured[0].prompt.split(marker, 1)[1]
    assert model.requests[0].output_schema['required'] == captured[0].output_schema['required']


def test_runtime_progress_is_optional_bounded_and_fails_closed():
    provider = ScriptedProvider([{'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'ok'}])
    chooser = RouterCandidateChooser(router=make_router(provider), task=TaskState(task_id='t', user='owner'),
        tools=(), max_input_tokens=8192,
        observation_progress=lambda: {'observed_sources': ['note'], 'unread_sources': ['other']})
    assert chooser('goal', 512)['kind'] == 'answer'
    assert '"observed_sources": ["note"]' in provider.requests[0].prompt
    chooser.observation_progress = lambda: {'observed_sources': ['same'], 'unread_sources': ['same']}
    with pytest.raises(ValueError, match='source_progress_unavailable'):
        chooser('goal', 512)
    assert len(provider.requests) == 1
