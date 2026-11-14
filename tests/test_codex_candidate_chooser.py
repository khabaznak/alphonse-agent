"""Synthetic subscription chooser tests; no Codex process or network."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceResult
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateHost, CandidateState, run_candidate
from alphonse.agent_v2.core.intelligence.v3.task_budget import (BudgetExhausted, BudgetLimits,
    SQLiteTaskBudgetStore, SyntheticEvaluationBudgetStore, TaskBudget)
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from alphonse.agent_v2.evaluation.codex_candidate_chooser import CodexSubscriptionChooser
from alphonse.agent_v2.evaluation.codex_subscription_provider import CodexSubscriptionProvider


def vec(actions=0, model_calls=0, input_tokens=0, output_tokens=0):
    return dict(actions=actions, model_calls=model_calls, input_tokens=input_tokens, output_tokens=output_tokens)


def make_case(tmp_path, store_type, *, allow_unknown):
    provider = CodexSubscriptionProvider(model='gpt-6-luna')
    events, calls, prompts = [], [], []
    decisions = iter([
        {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'index'}, 'text': None,
         'usage': {'input_tokens': 1, 'output_tokens': 1}},  # malicious model claim
        {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': '42'},
    ])
    def generate(request):
        assert request.purpose == InferencePurpose.SIMPLE_LOOP_ACTION
        prompts.append(request.prompt)
        value = next(decisions)
        import json
        return InferenceResult(json_value=value, raw_response=json.dumps(value), usage={})
    provider.generate_json = generate
    chooser = CodexSubscriptionChooser(provider=provider, model='gpt-6-luna',
        max_input_tokens=4096, emit=events.append)
    registry = InMemoryToolRegistry()
    descriptor = ToolDescriptor('native.fixture_read', 'fixture_read', ToolKind.NATIVE, read_only=True,
        argument_schema={'type': 'object', 'properties': {'source': {'type': 'string'}}, 'required': ['source']})
    registry.register(ToolDefinition(descriptor, lambda args: calls.append(args) or {'candidate_complete': True, 'value': '42'}))
    task = TaskState(task_id='task', user='fixture-user')
    ctx = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry)
    store = store_type(str(tmp_path / 'budget.db'))
    store.create('task', TaskBudget(BudgetLimits(vec(2, 4, 20000, 3000), vec(0, 1, 100, 100), 100000, 1000), started_at_ms=0))
    sources = {'goal': 'Read the number'}
    def record(actor, value):
        sources['ref'] = str(value['result'])
        return 'ref'
    host = CandidateHost(task, ctx, store, lambda _: None, chooser,
        lambda actor, ref: actor == 'fixture-user' and ref in sources,
        lambda actor, ref: sources[ref],
        lambda actor, tool, args: actor == 'fixture-user' and tool == 'native.fixture_read' and args == {'source': 'index'},
        record, lambda actor, text, refs: text == '42' and refs == ('ref',), lambda: 100,
        input_reservation=4096, output_reservation=512,
        synthetic_allow_unreported_usage=allow_unknown)
    return SimpleNamespace(host=host, state=CandidateState('task', 'fixture-user', 'goal'),
                           events=events, calls=calls, prompts=prompts)


def test_synthetic_loop_charges_reservations_and_preserves_unknown(tmp_path):
    case = make_case(tmp_path, SyntheticEvaluationBudgetStore, allow_unknown=True)
    result = run_candidate(case.state, case.host)
    assert result.status == 'completed' and result.response == '42'
    assert case.calls == [{'source': 'index'}] and len(case.prompts) == 2
    budget = case.host.budgets.load('task').usage()
    assert len(budget['unknown_usage']) == 2
    assert budget['charged']['input_tokens'] == 8192
    assert budget['charged']['output_tokens'] == 1024
    assert all(e['provider_usage'] == {'input_tokens': None, 'output_tokens': None} for e in case.events)
    assert all(e['usage_diagnostics']['input']['source'] == 'estimated' for e in case.events)


def test_default_path_still_blocks_on_unknown(tmp_path):
    case = make_case(tmp_path, SQLiteTaskBudgetStore, allow_unknown=False)
    result = run_candidate(case.state, case.host)
    assert result.status == 'blocked' and result.blocker == 'candidate_model_usage_unknown_or_overrun'
    assert not case.calls and len(case.host.budgets.load('task').usage()['unknown_usage']) == 1


def test_opt_in_rejects_ordinary_budget_store(tmp_path):
    case = make_case(tmp_path, SQLiteTaskBudgetStore, allow_unknown=True)
    result = run_candidate(case.state, case.host)
    assert result.blocker == 'candidate_synthetic_usage_policy_invalid'
    assert not case.prompts


def test_synthetic_store_still_blocks_inflight_and_enforces_limits(tmp_path):
    store = SyntheticEvaluationBudgetStore(str(tmp_path / 'budget.db'))
    store.create('task', TaskBudget(BudgetLimits(vec(1, 2, 100, 20), vec(), 10000, 0), started_at_ms=0))
    store.reserve('task', expected_revision=0, reservation_id='a', kind='model', amounts=vec(0, 1, 50, 10), now_ms=1)
    with pytest.raises(BudgetExhausted, match='budget_model_usage_unresolved'):
        store.reserve('task', expected_revision=1, reservation_id='b', kind='model', amounts=vec(0, 1, 50, 10), now_ms=2)
    budget = store.load('task')
    revision = budget.revision
    budget.settle('a', outcome='success', input_tokens=None, output_tokens=None)
    store.save('task', budget, expected_revision=revision)
    store.reserve('task', expected_revision=revision + 1, reservation_id='b', kind='model', amounts=vec(0, 1, 50, 10), now_ms=3)
    budget = store.load('task'); revision = budget.revision
    budget.settle('b', outcome='success', input_tokens=None, output_tokens=None)
    store.save('task', budget, expected_revision=revision)
    with pytest.raises(BudgetExhausted, match='budget_exhausted'):
        store.reserve('task', expected_revision=revision + 1, reservation_id='c', kind='model', amounts=vec(0, 1, 1, 1), now_ms=4)


def test_cancel_before_dispatch_and_oversized_output_fail_closed(tmp_path):
    case = make_case(tmp_path, SyntheticEvaluationBudgetStore, allow_unknown=True)
    chooser = case.host.choose
    chooser.cancellation_checker = lambda: True
    with pytest.raises(ValueError, match='cancelled_before'):
        chooser('goal', 512)
    assert not case.prompts
    chooser.cancellation_checker = None
    chooser.provider.generate_json = lambda request: InferenceResult(json_value={'kind':'answer','text':'x' * 100}, raw_response='x' * 100)
    with pytest.raises(ValueError, match='over_output_bound'):
        chooser('goal', 10)
    assert case.events[-1]['provider_usage']['output_tokens'] is None


def test_error_diagnostics_are_sanitized_and_invalid_usage_retained(tmp_path):
    case = make_case(tmp_path, SyntheticEvaluationBudgetStore, allow_unknown=True)
    chooser = case.host.choose
    def fail(_request):
        raise RuntimeError('secret-private-fixture')
    chooser.provider.generate_json = fail
    with pytest.raises(ValueError, match='candidate_codex_unavailable') as caught:
        chooser('goal', 512)
    assert caught.value.__context__ is None
    assert 'secret-private-fixture' not in str(case.events)
    chooser.provider.generate_json = lambda _: InferenceResult(json_value=[], usage={'input_tokens': 31})
    with pytest.raises(ValueError, match='invalid_decision'):
        chooser('goal', 512)
    event = case.events[-1]
    assert event['provider_usage'] == {'input_tokens': 31, 'output_tokens': None}
    assert event['usage_diagnostics']['input']['source'] == 'provider'
    assert event['usage_diagnostics']['output']['source'] == 'unknown'


def test_standard_store_rejects_synthetic_unknown_policy(tmp_path):
    store = SQLiteTaskBudgetStore(str(tmp_path / 'standard.db'))
    store.create('task', TaskBudget(BudgetLimits(vec(1, 2, 100, 20), vec(), 10000, 0), started_at_ms=0))
    with pytest.raises(ValueError, match='synthetic_budget_policy_not_available'):
        store.reserve('task', expected_revision=0, reservation_id='a', kind='model',
                      amounts=vec(0, 1, 50, 10), now_ms=1, _synthetic_allow_settled_unknown=True)
    assert store.load('task').revision == 0


def test_codex_receives_host_tool_manifest_and_dispatch_boundary(tmp_path):
    case = make_case(tmp_path, SyntheticEvaluationBudgetStore, allow_unknown=True)
    result = run_candidate(case.state, case.host)
    assert result.status == 'completed'
    first = case.prompts[0]
    assert 'Alphonse has the following host tool available' in first
    assert 'not a tool you need to call from Codex' in first
    assert '"tool_id": "native.fixture_read"' in first
    assert '"enum": ["index"]' in first
    assert 'Do not ask the user to enable an already listed tool' in first
    assert 'another available authorized source could materially change the conclusion' in first
    assert 'Inspect relevant sources when needed' in first
    assert case.calls == [{'source': 'index'}]


@pytest.mark.parametrize('invalid', [
    {'kind': 'answer', 'text': 'missing required fields'},
    {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': {'unexpected': 'object'}},
    {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'not-authorized'}, 'text': None},
])
def test_chooser_supplies_source_scoped_schema_and_never_coerces_invalid_action(tmp_path, invalid):
    case = make_case(tmp_path, SyntheticEvaluationBudgetStore, allow_unknown=True)
    requests = []
    def generate(request):
        requests.append(request)
        return InferenceResult(json_value=invalid, usage={})
    case.host.choose.provider.generate_json = generate
    result = case.host.choose('synthetic goal', 512)
    schema = requests[0].output_schema
    assert schema['required'] == ['kind', 'tool_id', 'arguments', 'text']
    assert schema['additionalProperties'] is False
    assert schema['properties']['arguments']['anyOf'][0]['properties']['source']['enum'] == ['index']
    assert result['kind'] == 'invalid' and 'text' not in result
    assert case.calls == []


def test_unconfigured_source_cannot_enter_tool_manifest(tmp_path):
    case = make_case(tmp_path, SyntheticEvaluationBudgetStore, allow_unknown=True)
    with pytest.raises(ValueError, match='invalid_fixture_sources'):
        CodexSubscriptionChooser(provider=case.host.choose.provider, model='gpt-6-luna',
            max_input_tokens=4096, emit=lambda _: None, allowed_source_names=('secret',))
