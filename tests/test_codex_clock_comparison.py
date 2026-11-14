"""Real-loop clock switch tests with deterministic provider doubles only."""
import json
from urllib.request import urlopen

import pytest

from alphonse.agent_v2.core.inference import InferenceRequest, InferenceResult, InferencePurpose
from alphonse.agent_v2.evaluation import codex_clock_comparison as comparison
from alphonse.agent_v2.evaluation.codex_subscription_provider import CodexSubscriptionProvider
from alphonse.agent_v2.evaluation.simplified_loop import Run

MODEL = 'gpt-6-luna'
ANSWER = {'local_time': '08:00', 'local_date': '2026-04-12', 'timezone': 'America/Mexico_City'}


def test_clock_answer_diagnostics_are_field_specific_and_sanitized():
    def diagnose(value):
        return comparison._clock_answer_diagnostics(value, expected_instant_utc='2026-04-12T14:00:00+00:00',
                                                    expected_timezone='America/Mexico_City')
    assert diagnose(json.dumps(ANSWER)) == {'format': 'object', 'exact_answer': True, 'extra_fields': False,
                                            'local_time': 'expected', 'local_date': 'expected', 'timezone': 'expected'}
    private = 'private-synthetic-secret'
    result = diagnose(json.dumps({'local_time': private, 'local_date': 12, private: private}))
    assert result == {'format': 'object', 'exact_answer': False, 'extra_fields': True,
                      'local_time': 'other', 'local_date': 'invalid_type', 'timezone': 'missing'}
    assert private not in repr(result)
    assert diagnose(private) == {'format': 'invalid_json'}
    assert diagnose('[]') == {'format': 'not_object'}
    assert diagnose(None) == {'format': 'no_answer'}


def test_candidate_actual_loop_runs_same_subscription_provider_and_checks_clock(monkeypatch):
    decisions = iter([
        {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'index'}, 'text': None},
        {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': json.dumps(ANSWER)},
        {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': json.dumps(ANSWER)},
    ])
    seen = []
    def fake_generate(self, request):
        seen.append(request)
        data = next(decisions)
        return InferenceResult(json_value=data, raw_response=json.dumps(data))
    monkeypatch.setattr(CodexSubscriptionProvider, 'generate_json', fake_generate)
    provider = CodexSubscriptionProvider(model=MODEL)
    result = comparison.run_clock_arm(arm='simple', provider=provider, model=MODEL)
    assert result['status'] == 'completed' and result['passed'] is True
    assert result['answer_diagnostics']['exact_answer'] is True
    assert len(seen) == 3 and all(item.model_profile.model == MODEL for item in seen)
    assert result['delivery'] == 'unassessed' and result['qualification'] == 'unassessed'
    events = [e for e in result['events'] if e['kind'] == 'system_two']
    assert len(events) == 3
    assert all(e['usage_diagnostics']['input']['source'] == 'estimated' for e in events)
    assert all(e['provider_usage']['input_tokens'] is None for e in events)
    assert result['metrics']['system_two_calls'] == 3
    assert result['metrics']['provider_input_tokens'] is None
    assert result['metrics']['input_tokens_measured_or_estimated'] > 0


def test_switch_runs_both_arms_with_same_provider_and_fresh_clock(monkeypatch):
    seen = []
    provider = CodexSubscriptionProvider(model=MODEL)
    def fake_v3(*, url, provider, jev, model, emit, cancelled, trace=None):
        seen.append((url, provider, jev, model))
        with urlopen(url, timeout=2) as response:
            json.load(response)
        return Run(status='completed', response=json.dumps(ANSWER))
    def fake_simple(*, url, provider, model, emit, cancelled, trace=None):
        seen.append((url, provider, None, model))
        with urlopen(url, timeout=2) as response:
            json.load(response)
        return Run(status='completed', response=json.dumps(ANSWER))
    monkeypatch.setattr(comparison, '_run_v3', fake_v3)
    monkeypatch.setattr(comparison, '_run_simple', fake_simple)
    result = comparison.compare_clock(provider=provider, model=MODEL, jev_factory=lambda: object())
    assert [item['arm'] for item in result['runs']] == ['v3', 'simple']
    assert all(item['passed'] for item in result['runs'])
    assert all(item[1] is provider and item[3] == MODEL for item in seen)
    assert seen[0][0] != seen[1][0]
    assert result['qualification'] == 'unassessed'


def test_non_subscription_and_invalid_arm_are_rejected_before_work():
    with pytest.raises(ValueError, match='matching_codex_subscription_provider_required'):
        comparison.run_clock_arm(arm='simple', provider=object(), model=MODEL)
    with pytest.raises(ValueError, match='unsupported_clock_arm'):
        comparison.run_clock_arm(arm='missing', provider=object(), model=MODEL)


def test_v3_provider_failure_records_unknown_output_not_fabricated_usage():
    class Failure:
        def generate_json(self, request):
            raise RuntimeError('secret diagnostic')
    events = []
    provider = comparison._TrackedProvider(Failure(), events.append)
    with pytest.raises(RuntimeError):
        provider.generate_json(InferenceRequest(prompt='synthetic prompt', purpose=InferencePurpose.TACTICAL_ACTION))
    assert events[0]['status'] == 'failed'
    assert events[0]['usage_diagnostics']['output']['source'] == 'unknown'
    assert 'secret diagnostic' not in repr(events)
    assert comparison._summarize_events(events)['output_tokens_measured_or_estimated'] is None


def test_v3_actual_loop_with_scripted_codex_provider_is_fixture_plumbing_only(monkeypatch, tmp_path):
    from alphonse.agent_v2.evaluation.deterministic_adapters import (
        _Action, _DeterministicProvider, _ReplayJev, _RunState, _Scenario,
    )
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
    from alphonse.agent_v2.core.tools.registry.native.respond import build_respond_tool_definition
    installed = []
    original_register = InMemoryToolRegistry.register
    def inspect_register(registry, definition):
        if definition.descriptor.tool_id == 'native.respond':
            installed.append(definition)
        return original_register(registry, definition)
    monkeypatch.setattr(InMemoryToolRegistry, 'register', inspect_register)
    action = _Action('native.fixture_read', 'project_record_search', {'source': 'index'}, role='read_clock', behavior='read')
    replay = _RunState(_Scenario((action,), 'Answer from the synthetic clock.', json.dumps(ANSWER)), tmp_path, 'hierarchical_v3')
    delegate = _DeterministicProvider(replay)
    def json_method(self, request):
        if request.purpose == InferencePurpose.TASK_UNDERSTANDING:
            return InferenceResult(json_value={'requested_outcome': comparison.GOAL, 'supplied_parameters': [],
                'constraints': [], 'unresolved_dependencies': [], 'superseded_constraints': []})
        return delegate.generate_json(request)
    monkeypatch.setattr(CodexSubscriptionProvider, 'generate_json', json_method)
    monkeypatch.setattr(CodexSubscriptionProvider, 'generate_markdown', lambda self, request: delegate.generate_markdown(request))
    monkeypatch.setattr(CodexSubscriptionProvider, 'plan_tool_call', lambda self, request: delegate.plan_tool_call(request))
    provider = CodexSubscriptionProvider(model=MODEL)
    trace = []
    result = comparison.run_clock_arm(arm='v3', provider=provider, model=MODEL, jev=_ReplayJev(),
                                      private_trace=trace.append)
    actual = build_respond_tool_definition()
    assert installed[-1].descriptor == actual.descriptor
    assert installed[-1].argument_schema == actual.argument_schema
    assert installed[-1].descriptor.read_only is False
    assert installed[-1].descriptor.metadata['side_effect_class'] == 'user_response'
    assert result['passed'] is True, result
    assert result['oracle']['clock_source_matches'] is True
    assert any(event['kind'] == 'system_two' for event in result['events'])
    assert {'event': 'source_read_attempt', 'arm': 'v3', 'source': 'index'} in trace
    assert {'event': 'source_read_observed', 'arm': 'v3', 'source': 'index'} in trace
    assert {'event': 'response_prepared', 'arm': 'v3'} in trace
    assert trace[-1]['event'] == 'terminal' and trace[-1]['response_accepted'] is True
    assert json.dumps(ANSWER) not in repr(trace)
    # A scripted provider is not evidence of real task effectiveness.
    assert result['qualification'] == 'unassessed'


def test_reason_category_never_exposes_unrecognized_reason():
    assert comparison._reason_category('candidate_answer_unsupported') == 'evidence_unavailable'
    assert comparison._reason_category('secret reason') == 'other'


def test_each_failed_arm_is_preserved_and_error_text_not_exposed(monkeypatch):
    provider = CodexSubscriptionProvider(model=MODEL)
    def fail(**_kwargs):
        raise ValueError('sensitive-contents')
    monkeypatch.setattr(comparison, 'run_clock_arm', fail)
    result = comparison.compare_clock(provider=provider, model=MODEL, jev_factory=lambda: object())
    assert [x['arm'] for x in result['runs']] == ['v3', 'simple']
    assert all(x['passed'] is None and x['error_type'] == 'ValueError' for x in result['runs'])
    assert 'sensitive-contents' not in repr(result)


def test_opt_in_trace_categorizes_question_without_exposing_model_text(monkeypatch):
    secret = 'synthetic-sensitive-value'
    def fake_generate(self, request):
        data = {'kind': 'question', 'tool_id': None, 'arguments': None, 'text': 'Please enable the unavailable tool ' + secret}
        return InferenceResult(json_value=data, raw_response=json.dumps(data))
    monkeypatch.setattr(CodexSubscriptionProvider, 'generate_json', fake_generate)
    observed = []
    result = comparison.run_clock_arm(arm='simple', provider=CodexSubscriptionProvider(model=MODEL),
                                      model=MODEL, private_trace=observed.append)
    assert result['status'] == 'waiting_user'
    assert observed == [{'event': 'model_decision', 'arm': 'simple', 'kind': 'question',
                         'tool': None, 'source': None, 'after_answer_proposal': False,
                         'mentions_tool_availability': True}]
    assert secret not in repr(observed) and secret not in repr(result)


def test_candidate_trace_records_read_order_without_source_contents(monkeypatch):
    secret = 'synthetic-private-source-content'
    decisions = iter([
        {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'conversation'}, 'text': None},
        {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'initial response'},
        {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'treatment-status'}, 'text': None},
        {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'final response'},
    ])
    def fake_generate(self, request):
        data = next(decisions)
        return InferenceResult(json_value=data, raw_response=json.dumps(data))
    class FakeSource:
        def __init__(self, url, *, allowed_source_names):
            pass
        def read(self, args):
            return {'source': args['source'], 'content': {'value': secret}, 'candidate_complete': True}
    monkeypatch.setattr(CodexSubscriptionProvider, 'generate_json', fake_generate)
    monkeypatch.setattr(comparison, 'LocalSource', FakeSource)
    observed = []
    outcome = comparison._run_simple(url='fixture', provider=CodexSubscriptionProvider(model=MODEL),
        model=MODEL, emit=lambda _: None, cancelled=lambda: False, trace=observed.append,
        allowed_source_names=('conversation', 'treatment-status'), required_source_names=None,
        verification_mode='matched_provenance')
    assert outcome.status == 'completed'
    assert [(e['event'], e.get('source'), e.get('after_answer_proposal')) for e in observed] == [
        ('model_decision', 'conversation', False), ('source_read_observed', 'conversation', None),
        ('model_decision', None, False), ('model_decision', 'treatment-status', True),
        ('source_read_observed', 'treatment-status', None), ('model_decision', None, True),
    ]
    assert secret not in repr(observed) and 'initial response' not in repr(observed)


def test_source_progress_reflects_only_recorded_reads_without_oracle_hints(monkeypatch):
    decisions = iter([
        {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'conversation'}, 'text': None},
        {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'done'},
        {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'done'},
    ])
    prompts = []
    def fake_generate(self, request):
        prompts.append(request.prompt)
        data = next(decisions)
        return InferenceResult(json_value=data, raw_response=json.dumps(data))
    class FakeSource:
        def __init__(self, url, *, allowed_source_names):
            pass
        def read(self, args):
            return {'source': args['source'], 'content': {'value': 'fixture'}, 'candidate_complete': True}
    monkeypatch.setattr(CodexSubscriptionProvider, 'generate_json', fake_generate)
    monkeypatch.setattr(comparison, 'LocalSource', FakeSource)
    outcome = comparison._run_simple(url='fixture', provider=CodexSubscriptionProvider(model=MODEL),
        model=MODEL, emit=lambda _: None, cancelled=lambda: False,
        allowed_source_names=('conversation', 'prescription'), required_source_names=None,
        verification_mode='matched_provenance', show_source_progress=True)
    assert outcome.status == 'completed'
    assert '"observed_sources": [], "unread_sources": ["conversation", "prescription"]' in prompts[0]
    assert '"observed_sources": ["conversation"], "unread_sources": ["prescription"]' in prompts[1]
    assert 'not a list of required sources or a judgment of relevance' in prompts[1]
    assert len(prompts) == 3  # pre-answer review does not force reading every source


def test_trace_callback_failure_does_not_change_clock_result(monkeypatch):
    decisions = iter([
        {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'index'}, 'text': None},
        {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': json.dumps(ANSWER)},
        {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': json.dumps(ANSWER)},
    ])
    def fake_generate(self, request):
        data = next(decisions)
        return InferenceResult(json_value=data, raw_response=json.dumps(data))
    monkeypatch.setattr(CodexSubscriptionProvider, 'generate_json', fake_generate)
    def broken(_event):
        raise RuntimeError('test diagnostic failure')
    result = comparison.run_clock_arm(arm='simple', provider=CodexSubscriptionProvider(model=MODEL),
                                      model=MODEL, private_trace=broken)
    assert result['passed'] is True


def test_private_trace_handles_malformed_model_tool_without_affecting_result():
    class Malformed:
        def generate_json(self, request):
            return InferenceResult(json_value={'tool_id': ['invalid']})
    traces = []
    provider = comparison._TrackedProvider(Malformed(), lambda _: None, traces.append)
    result = provider.generate_json(InferenceRequest(prompt='synthetic', purpose=InferencePurpose.TACTICAL_ACTION))
    assert result.json_value == {'tool_id': ['invalid']}
    assert traces[0]['tool'] is None


def test_candidate_recovery_uses_advertised_inventory_not_hidden_required_set(monkeypatch):
    from alphonse.agent_v2.core.intelligence.v3 import single_loop_candidate
    observed = []
    def inspect(state, host):
        # Same visible inventory and read history, different hidden requirements.
        refs = ('read:index', 'read:conversation')
        observed.append((host.verify_response(comparison.ACTOR, 'answer', refs),
                         host.remaining_advertised_evidence(comparison.ACTOR, refs),
                         host.remaining_advertised_evidence(comparison.ACTOR, refs + ('read:prescription',)),
                         host.remaining_advertised_evidence('other-user', refs)))
        state.status = 'blocked'
        return state
    monkeypatch.setattr(single_loop_candidate, 'run_candidate', inspect)
    provider = CodexSubscriptionProvider(model=MODEL)
    for required in (('index', 'conversation'), ('conversation', 'prescription')):
        comparison._run_simple(url='http://127.0.0.1:1/' + 'x' * 43 + '/', provider=provider, model=MODEL,
            emit=lambda _: None, cancelled=lambda: False, allowed_source_names=('index', 'conversation', 'prescription'),
            required_source_names=required)
    # Verification uses required sources, but available-work eligibility does not.
    assert [row[1:] for row in observed] == [(True, False, False), (True, False, False)]


def test_matched_provenance_rejects_oracle_key_before_running():
    with pytest.raises(ValueError, match='matched_mode_disallows_required_sources'):
        comparison._run_simple(url='http://127.0.0.1:1/' + 'x' * 43 + '/',
            provider=CodexSubscriptionProvider(model=MODEL), model=MODEL,
            emit=lambda _: None, cancelled=lambda: False,
            allowed_source_names=('index', 'prescription'),
            required_source_names=('prescription',), verification_mode='matched_provenance')
