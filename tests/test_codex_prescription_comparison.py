"""Processor comparison plumbing; fake provider only, no clinical data."""
import json
from urllib.request import urlopen

import pytest

from alphonse.agent_v2.core.inference import InferenceResult
from alphonse.agent_v2.evaluation import codex_prescription_comparison as comparison
from alphonse.agent_v2.evaluation.codex_subscription_provider import CodexSubscriptionProvider
from alphonse.agent_v2.evaluation.simplified_loop import Run

MODEL = 'gpt-6-luna'
ANSWER, EXPECTED_SOURCES = comparison._expected()


def test_simple_real_processor_reads_discoverable_sources_and_independent_oracle(monkeypatch):
    decisions = iter([
        *({'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': name}}
          for name in ('index', 'conversation', 'prescription', 'treatment-status')),
        {'kind': 'answer', 'text': json.dumps(ANSWER)},
    ])
    observed = []
    def fake_provider(self, request):
        observed.append(request)
        value = next(decisions)
        return InferenceResult(json_value=value, raw_response=json.dumps(value))
    monkeypatch.setattr(CodexSubscriptionProvider, 'generate_json', fake_provider)
    result = comparison.run_prescription_arm(arm='simple', provider=CodexSubscriptionProvider(model=MODEL), model=MODEL)
    assert result['status'] == 'completed' and result['passed'] is True
    assert set(result['oracle']['observed_required_sources']) == set(EXPECTED_SOURCES)
    assert len(observed) == 5 and all(request.model_profile.model == MODEL for request in observed)
    assert result['metrics']['provider_input_tokens'] is None
    assert result['delivery'] == result['qualification'] == 'unassessed'


def test_guess_without_reads_fails_even_when_answer_correct(monkeypatch):
    def fake_simple(**kwargs):
        return Run(status='completed', response=json.dumps(ANSWER))
    monkeypatch.setattr(comparison, '_run_simple', fake_simple)
    result = comparison.run_prescription_arm(arm='simple', provider=CodexSubscriptionProvider(model=MODEL), model=MODEL)
    assert result['passed'] is False
    assert result['oracle']['answer_matches_key'] is True
    assert result['oracle']['observed_required_sources'] == []


@pytest.mark.parametrize(('reason', 'category'), [
    ('candidate_answer_unsupported', 'evidence_unavailable'),
    ('cancelled', 'cancelled'),
    ('secret token /private/path', 'other'),
    (None, None),
])
def test_terminal_reasons_are_closed_set_and_do_not_expose_details(monkeypatch, reason, category):
    monkeypatch.setattr(comparison, '_run_simple',
                        lambda **kwargs: Run(status='blocked', response='', reason=reason))
    result = comparison.run_prescription_arm(
        arm='simple', provider=CodexSubscriptionProvider(model=MODEL), model=MODEL)
    assert result['status'] == 'blocked' and result['passed'] is False
    assert result['reason_category'] == category
    assert 'secret token /private/path' not in repr(result)


def test_wrong_dose_fails_after_sources_are_read(monkeypatch):
    def fake_simple(*, url, **_kwargs):
        for suffix in ('', 'conversation', 'prescription', 'treatment-status'):
            with urlopen(url + suffix, timeout=2) as response:
                json.load(response)
        return Run(status='completed', response=json.dumps({'status':'schedule','medications':[{'name':'Asterin','dose':'20 mg','time':'08:00'}]}))
    monkeypatch.setattr(comparison, '_run_simple', fake_simple)
    result = comparison.run_prescription_arm(arm='simple', provider=CodexSubscriptionProvider(model=MODEL), model=MODEL)
    assert result['passed'] is False and result['oracle']['answer_matches_key'] is False
    assert set(result['oracle']['observed_required_sources']) == set(EXPECTED_SOURCES)


def test_two_arms_use_same_provider_with_fresh_sources_and_keep_errors(monkeypatch):
    seen = []
    provider = CodexSubscriptionProvider(model=MODEL)
    def fake_v3(*, url, provider, jev, model, **_kwargs):
        seen.append((url, provider, model, jev))
        raise ValueError('must not expose this')
    def fake_simple(*, url, provider, model, **_kwargs):
        seen.append((url, provider, model, None))
        for suffix in ('', 'conversation', 'prescription', 'treatment-status'):
            with urlopen(url + suffix, timeout=2) as response:
                json.load(response)
        return Run(status='completed', response=json.dumps(ANSWER))
    monkeypatch.setattr(comparison, '_run_v3', fake_v3)
    monkeypatch.setattr(comparison, '_run_simple', fake_simple)
    report = comparison.compare_prescription(provider=provider, model=MODEL, jev_factory=lambda: object())
    assert [row['arm'] for row in report['runs']] == ['v3', 'simple']
    assert report['runs'][0]['status'] == 'error' and report['runs'][0]['passed'] is False
    assert report['runs'][1]['passed'] is True
    assert seen[0][0] != seen[1][0] and all(row[1] is provider and row[2] == MODEL for row in seen)
    assert 'must not expose this' not in repr(report)


def test_invalid_provider_or_arm_does_not_run():
    with pytest.raises(ValueError, match='unsupported_prescription_arm'):
        comparison.run_prescription_arm(arm='other', provider=object(), model=MODEL)
    with pytest.raises(ValueError, match='matching_codex_subscription_provider_required'):
        comparison.run_prescription_arm(arm='simple', provider=object(), model=MODEL)


def test_v3_real_processor_runs_synthetic_path_with_provider_double(monkeypatch, tmp_path):
    from alphonse.agent_v2.core.inference import InferencePurpose
    from alphonse.agent_v2.evaluation.deterministic_adapters import _Action, _DeterministicProvider, _ReplayJev, _RunState, _Scenario
    actions = tuple(_Action('native.fixture_read', 'project_record_search', {'source': name}, role=name, behavior='read')
                    for name in ('index', 'conversation', 'prescription', 'treatment-status'))
    replay = _RunState(_Scenario(actions, 'Answer from synthetic records.', json.dumps(ANSWER)), tmp_path, 'hierarchical_v3')
    delegate = _DeterministicProvider(replay)
    def json_method(self, request):
        if request.purpose == InferencePurpose.TASK_UNDERSTANDING:
            return InferenceResult(json_value={'requested_outcome': comparison._GOAL, 'supplied_parameters': [],
                'constraints': [], 'unresolved_dependencies': [], 'superseded_constraints': []})
        return delegate.generate_json(request)
    monkeypatch.setattr(CodexSubscriptionProvider, 'generate_json', json_method)
    monkeypatch.setattr(CodexSubscriptionProvider, 'generate_markdown', lambda self, request: delegate.generate_markdown(request))
    monkeypatch.setattr(CodexSubscriptionProvider, 'plan_tool_call', lambda self, request: delegate.plan_tool_call(request))
    result = comparison.run_prescription_arm(arm='v3', provider=CodexSubscriptionProvider(model=MODEL), model=MODEL, jev=_ReplayJev())
    assert result['passed'] is True, result
    assert set(result['oracle']['observed_required_sources']) == set(EXPECTED_SOURCES)
    assert result['qualification'] == 'unassessed'


def test_natural_question_does_not_force_redundant_index_or_source_hint(monkeypatch):
    assert 'find the prescription' not in comparison._GOAL.lower()
    assert 'treatment-start' not in comparison._GOAL.lower()
    decisions = iter([
        *({'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': name}}
          for name in ('conversation', 'prescription', 'treatment-status')),
        {'kind': 'answer', 'text': json.dumps(ANSWER)},
    ])
    def fake_provider(self, request):
        value = next(decisions)
        return InferenceResult(json_value=value, raw_response=json.dumps(value))
    monkeypatch.setattr(CodexSubscriptionProvider, 'generate_json', fake_provider)
    result = comparison.run_prescription_arm(arm='simple', provider=CodexSubscriptionProvider(model=MODEL), model=MODEL)
    assert result['status'] == 'completed' and result['passed'] is True
    assert set(result['oracle']['observed_required_sources']) == {'conversation', 'prescription', 'treatment-status'}
    assert result['oracle']['required_sources_for_task'] == ['conversation', 'prescription', 'treatment-status']


def test_early_answer_recovers_without_disclosing_oracle(monkeypatch):
    decisions = iter([
        {'kind': 'answer', 'text': '{}'},
        *({'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': name}}
          for name in ('conversation', 'prescription', 'treatment-status')),
        {'kind': 'answer', 'text': json.dumps(ANSWER)},
    ])
    prompts = []
    def fake_provider(self, request):
        prompts.append(request.prompt)
        value = next(decisions)
        return InferenceResult(json_value=value, raw_response=json.dumps(value))
    monkeypatch.setattr(CodexSubscriptionProvider, 'generate_json', fake_provider)
    result = comparison.run_prescription_arm(arm='simple', provider=CodexSubscriptionProvider(model=MODEL), model=MODEL)
    assert result['passed'] is True
    assert 'The proposed answer is not yet verified' in prompts[1]
    assert json.dumps(ANSWER) not in prompts[1]
    feedback = prompts[1].split('The proposed answer is not yet verified', 1)[1]
    assert 'conversation' not in feedback and 'treatment-status' not in feedback
