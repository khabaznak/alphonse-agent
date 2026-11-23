from copy import deepcopy

import pytest

from alphonse.agent_v2.core.intelligence.simple_loop.owner_progress_review import review_progress, window_evidence_refs
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_question import render_window_question
from alphonse.agent_v2.system_one import JevCriterionDecisionProvider


OBSERVATIONS = [{'tool': 'native.web_search', 'arguments': {'query': 'synthetic report'},
                 'observation': {'status': 'success', 'results': ['synthetic result']}}]


@pytest.mark.parametrize('probability,expected', [(.9, 'progress'), (.1, 'no_progress'), (.5, 'unknown'),
                                                  (None, 'unknown'), (float('nan'), 'unknown'), (True, 'unknown')])
def test_progress_probability_is_not_authority(probability, expected):
    calls = []
    result = review_progress(request='Find a synthetic report', observations=deepcopy(OBSERVATIONS),
        authorize_disclosure=lambda: True, judge=lambda **kw: calls.append(kw) or probability,
        no_threshold=.3, yes_threshold=.7)
    assert result == expected
    assert calls[0]['state']['observations'] == OBSERVATIONS


def test_disclosure_denial_and_incomplete_evidence_never_call_provider():
    def fail(**_):
        pytest.fail('provider called without disclosure or complete evidence')
    assert review_progress(request='task', observations=OBSERVATIONS, authorize_disclosure=lambda: False,
        judge=fail, no_threshold=.3, yes_threshold=.7) == 'unknown'
    assert review_progress(request='task', observations=[], authorize_disclosure=lambda: True,
        judge=fail, no_threshold=.3, yes_threshold=.7) == 'unknown'
    too_large = deepcopy(OBSERVATIONS)
    too_large[0]['observation']['payload'] = 'x' * 33000
    assert review_progress(request='task', observations=too_large, authorize_disclosure=lambda: True,
        judge=fail, no_threshold=.3, yes_threshold=.7) == 'unknown'


def test_only_new_window_evidence_is_reviewed_and_legacy_baseline_is_unknown():
    assert window_evidence_refs(['old', 'new'], {'old'}) == ('new',)
    assert window_evidence_refs(['old'], {'old'}) == ()
    assert window_evidence_refs(['old', 'new'], None) == ()
    assert window_evidence_refs(['new'], {'old'}) == ()


def test_provider_error_is_unknown():
    def unavailable(**_):
        raise RuntimeError('provider unavailable')
    assert review_progress(request='task', observations=OBSERVATIONS, authorize_disclosure=lambda: True,
        judge=unavailable, no_threshold=.3, yes_threshold=.7) == 'unknown'


def test_disclosure_revocation_during_call_discards_assessment():
    decisions = iter([True, False])
    assert review_progress(request='task', observations=OBSERVATIONS,
        authorize_disclosure=lambda: next(decisions), judge=lambda **_: .95,
        no_threshold=.3, yes_threshold=.7) == 'unknown'


def test_jev_receives_atomic_question_and_returns_probability():
    provider = JevCriterionDecisionProvider.__new__(JevCriterionDecisionProvider)
    seen = []
    provider._evaluate = lambda **kw: seen.append(kw) or {'answers': {'observed_progress': {'type': 'noul', 'noul': .81}}}
    assert provider.judge_observed_progress(state={'request': 'r', 'observations': OBSERVATIONS}) == .81
    assert set(seen[0]['questions']) == {'observed_progress'}


def test_question_distinguishes_assessment_and_keeps_v1_unchanged():
    pause = {'schema_version': 2, 'counter': 2, 'metric': 'actions', 'progress_assessment': 'progress'}
    window = {'increment': {'actions': 2, 'model_calls': 4, 'input_tokens': 10, 'output_tokens': 10}, 'elapsed_increment_ms': 5000}
    assert render_window_question(pause, window).startswith('A preliminary review found evidence')
    pause['schema_version'] = 1
    assert render_window_question(pause, window).startswith('This request has reached')
