"""Offline integration contract; no test calls a model provider."""
import pytest

from alphonse.agent_v2.core.inference import InferenceResult
from alphonse.agent_v2.evaluation.model_candidate_bridge import BoundedV3Provider, CandidateDecisionBridge


class Transport:
    offline = True

    def __init__(self, result):
        self.result = result
        self.calls = []

    def generate_json(self, request, *, max_input_tokens, max_output_tokens):
        self.calls.append((request, max_input_tokens, max_output_tokens))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_provider_usage_is_observed_not_model_supplied():
    transport = Transport(InferenceResult(json_value={'kind': 'answer', 'text': 'synthetic',
        'usage': {'input_tokens': 1, 'output_tokens': 1}},
        usage={'input_tokens': 41, 'output_tokens': 7}))
    events = []
    chooser = CandidateDecisionBridge(transport, 2048, events.append)
    result = chooser('fixture observation', 128)
    assert result['usage'] == {'input_tokens': 41, 'output_tokens': 7}
    assert transport.calls[0][0].prompt.endswith('fixture observation')
    assert transport.calls[0][1:] == (2048, 128)
    assert events[0]['provider_usage'] == result['usage']
    assert 'fixture observation' not in str(events)


def test_missing_usage_emits_separate_estimate_without_settling_budget():
    events = []
    raw = '{"kind":"answer","text":"hola"}'
    chooser = CandidateDecisionBridge(Transport(InferenceResult(
        json_value={'kind': 'answer', 'text': 'hola'}, raw_response=raw)), 2048,
        events.append, estimate_method='words', estimate_tokens_per_word=2,
        estimate_calibration='synthetic-es')
    result = chooser('Hola Alphonse', 128)
    assert result['usage'] == {'input_tokens': None, 'output_tokens': None}
    diagnostic = events[0]['usage_diagnostics']
    assert diagnostic['input']['source'] == diagnostic['output']['source'] == 'estimated'
    assert diagnostic['calibration'] == 'synthetic-es'
    assert 'Hola Alphonse' not in str(events) and raw not in str(events)


def test_no_observed_output_does_not_invent_output_estimate():
    events = []
    chooser = CandidateDecisionBridge(Transport(InferenceResult(
        json_value={'kind': 'answer', 'text': 'hola'})), 2048, events.append)
    chooser('Hola', 128)
    assert events[0]['usage_diagnostics']['input']['source'] == 'estimated'
    assert events[0]['usage_diagnostics']['output']['source'] == 'unknown'


@pytest.mark.parametrize('usage', [{}, {'input_tokens': True, 'output_tokens': -1},
                                  {'input_tokens': 5, 'output_tokens': None}])
def test_missing_or_invalid_provider_usage_is_unknown(usage):
    transport = Transport(InferenceResult(json_value={'kind': 'answer', 'text': 'x'}, usage=usage))
    events = []
    result = CandidateDecisionBridge(transport, 2048, events.append)('synthetic', 128)
    for key in ('input_tokens', 'output_tokens'):
        value = usage.get(key)
        assert result['usage'][key] == (value if type(value) is int and value >= 0 else None)


def test_rejects_nonoffline_transport_before_any_call():
    transport = Transport(InferenceResult())
    transport.offline = False
    with pytest.raises(ValueError, match='live_transport_not_authorized'):
        CandidateDecisionBridge(transport, 2048, lambda _: None)
    assert not transport.calls


def test_limit_includes_bridge_instructions_and_rejects_before_call():
    transport = Transport(InferenceResult())
    with pytest.raises(ValueError, match='candidate_projection_exceeds_input_limit'):
        CandidateDecisionBridge(transport, 12, lambda _: None)('x', 128)
    assert not transport.calls


def test_source_guidance_matches_declared_clock_inventory():
    transport = Transport(InferenceResult(json_value={'kind': 'answer', 'text': 'x'}, usage={}))
    chooser = CandidateDecisionBridge(transport, 2048, lambda _: None,
        fixture_argument='source', allowed_source_names=('index',))
    chooser('synthetic clock question', 128)
    prompt = transport.calls[0][0].prompt
    assert 'Available synthetic sources: ["index"]' in prompt
    assert 'prescription' not in prompt and 'treatment-status' not in prompt
    assert '"arguments":{"source":"..."}' in prompt


@pytest.mark.parametrize('names', [(), ['index'], ('index', 'index'), ('private',), (None,)])
def test_source_guidance_rejects_invalid_inventory(names):
    with pytest.raises(ValueError, match='invalid_fixture_sources'):
        CandidateDecisionBridge(Transport(InferenceResult()), 2048, lambda _: None,
                                fixture_argument='source', allowed_source_names=names)


@pytest.mark.parametrize('argument', [[], {}, None, True])
def test_invalid_fixture_argument_has_sanitized_error(argument):
    with pytest.raises(ValueError, match='invalid_fixture_argument'):
        CandidateDecisionBridge(Transport(InferenceResult()), 2048, lambda _: None,
                                fixture_argument=argument)


def test_framed_prompt_at_boundary_is_accepted_and_one_byte_over_is_rejected():
    from alphonse.agent_v2.evaluation.model_candidate_bridge import _INSTRUCTIONS
    transport = Transport(InferenceResult(json_value={'kind': 'answer', 'text': 'x'}, usage={}))
    limit = len((_INSTRUCTIONS + 'x').encode('utf-8'))
    chooser = CandidateDecisionBridge(transport, limit, lambda _: None)
    chooser('x', 128)
    with pytest.raises(ValueError, match='candidate_projection_exceeds_input_limit'):
        chooser('xy', 128)
    assert len(transport.calls) == 1


def test_failure_records_unknown_without_error_or_prompt():
    events = []
    chooser = CandidateDecisionBridge(Transport(RuntimeError('private error')), 2048, events.append)
    with pytest.raises(RuntimeError):
        chooser('private prompt', 128)
    assert events[0]['status'] == 'failed' and events[0]['provider_usage'] is None
    assert 'private' not in str(events)


def test_actual_candidate_refuses_to_finish_when_usage_is_unknown(tmp_path):
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateHost, CandidateState, run_candidate
    from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits, SQLiteTaskBudgetStore, TaskBudget
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry

    def units(actions=0, calls=0, inp=0, out=0):
        return {'actions': actions, 'model_calls': calls, 'input_tokens': inp, 'output_tokens': out}

    store = SQLiteTaskBudgetStore(str(tmp_path / 'budget.db'))
    store.create('task', TaskBudget(BudgetLimits(units(1, 3, 5000, 500),
                 units(0, 1, 100, 50), 100000, 1000), started_at_ms=0))
    task = TaskState(task_id='task', user='actor')
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=InMemoryToolRegistry())
    chooser = CandidateDecisionBridge(Transport(InferenceResult(
        json_value={'kind': 'answer', 'text': 'synthetic'}, usage={})), 1024, lambda _: None)
    host = CandidateHost(task, context, store, lambda _: None, chooser,
        lambda actor, ref: True, lambda actor, ref: 'synthetic prompt',
        lambda actor, tool, args: False, lambda actor, result: '',
        lambda actor, text, refs: True, lambda: 100,
        input_reservation=1024, output_reservation=128, projection_budget=512)
    state = run_candidate(CandidateState('task', 'actor', 'goal'), host)
    assert state.status == 'blocked'
    assert state.blocker == 'candidate_model_usage_unknown_or_overrun'


def test_candidate_rejects_bridge_cap_exceeding_durable_reservation_before_dispatch(tmp_path):
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateHost, CandidateState, run_candidate
    from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits, SQLiteTaskBudgetStore, TaskBudget
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
    def units(actions=0, calls=0, inp=0, out=0):
        return {'actions': actions, 'model_calls': calls, 'input_tokens': inp, 'output_tokens': out}
    store = SQLiteTaskBudgetStore(str(tmp_path / 'budget.db'))
    store.create('task', TaskBudget(BudgetLimits(units(1, 3, 5000, 500),
                 units(0, 1, 100, 50), 100000, 1000), started_at_ms=0))
    transport = Transport(InferenceResult(json_value={'kind': 'answer', 'text': 'synthetic'},
                                           usage={'input_tokens': 10, 'output_tokens': 2}))
    chooser = CandidateDecisionBridge(transport, 2048, lambda _: None)
    host = CandidateHost(TaskState(task_id='task', user='actor'),
        CoreLoopContext(messages=InMemoryMessageQueue(), tools=InMemoryToolRegistry()),
        store, lambda _: None, chooser, lambda actor, ref: True,
        lambda actor, ref: 'synthetic', lambda actor, tool, args: False,
        lambda actor, result: '', lambda actor, text, refs: True, lambda: 100,
        input_reservation=1024, output_reservation=128, projection_budget=512)
    state = run_candidate(CandidateState('task', 'actor', 'goal'), host)
    assert state.status == 'blocked'
    assert state.blocker == 'candidate_chooser_input_cap_exceeds_reservation'
    assert not transport.calls


@pytest.mark.parametrize('cancel_stage', ['before', 'after'])
def test_candidate_honors_cancellation_around_provider_dispatch(tmp_path, cancel_stage):
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateHost, CandidateState, run_candidate
    from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits, SQLiteTaskBudgetStore, TaskBudget
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
    def units(actions=0, calls=0, inp=0, out=0):
        return {'actions': actions, 'model_calls': calls, 'input_tokens': inp, 'output_tokens': out}
    cancelled = {'value': cancel_stage == 'before'}
    class CancellingTransport(Transport):
        def generate_json(self, request, **kwargs):
            result = super().generate_json(request, **kwargs)
            cancelled['value'] = True
            return result
    transport = CancellingTransport(InferenceResult(json_value={'kind': 'answer', 'text': 'synthetic'},
                                                    usage={'input_tokens': 10, 'output_tokens': 2}))
    store = SQLiteTaskBudgetStore(str(tmp_path / 'budget.db'))
    store.create('task', TaskBudget(BudgetLimits(units(1, 3, 5000, 500),
                 units(0, 1, 100, 50), 100000, 1000), started_at_ms=0))
    host = CandidateHost(TaskState(task_id='task', user='actor'),
        CoreLoopContext(messages=InMemoryMessageQueue(), tools=InMemoryToolRegistry(),
                        cancellation_checker=lambda: cancelled['value']),
        store, lambda _: None, CandidateDecisionBridge(transport, 1024, lambda _: None),
        lambda actor, ref: True, lambda actor, ref: 'synthetic',
        lambda actor, tool, args: False, lambda actor, result: '',
        lambda actor, text, refs: True, lambda: 100,
        input_reservation=1024, output_reservation=128, projection_budget=512)
    state = run_candidate(CandidateState('task', 'actor', 'goal'), host)
    assert state.status == 'cancelled'
    assert len(transport.calls) == (0 if cancel_stage == 'before' else 1)
    assert not state.response


def test_v3_router_can_use_same_offline_transport_and_report_real_usage():
    from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest, InferenceRouter, ModelProfile
    transport = Transport(InferenceResult(json_value={'answer': 'value'},
                                         usage={'input_tokens': 20, 'output_tokens': 3}))
    events = []
    provider = BoundedV3Provider(transport, 2048, 128, events.append)
    router = InferenceRouter(provider=provider, default_profile=ModelProfile(
        provider='fixture', model='fixture', profile_id='fixture'))
    result = router.generate_json(InferenceRequest(prompt='synthetic', purpose=InferencePurpose.TACTICAL_ACTION))
    assert result.json_value == {'answer': 'value'}
    assert events[0]['provider_usage'] == {'input_tokens': 20, 'output_tokens': 3}
    assert transport.calls[0][1:] == (2048, 128)


def test_v3_provider_rejects_reported_overrun_and_blocks_live_transport():
    from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest
    transport = Transport(InferenceResult(json_value={}, usage={'input_tokens': 3000, 'output_tokens': 3}))
    events = []
    provider = BoundedV3Provider(transport, 2048, 128, events.append)
    with pytest.raises(ValueError, match='provider_reported_usage_over_limit'):
        provider.generate_json(InferenceRequest(prompt='synthetic', purpose=InferencePurpose.TACTICAL_ACTION))
    assert events[0]['provider_usage']['input_tokens'] == 3000
    transport.offline = False
    with pytest.raises(ValueError, match='live_transport_not_authorized'):
        BoundedV3Provider(transport, 2048, 128, events.append)
