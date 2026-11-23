import hashlib

import requests

from alphonse.agent_v2.evaluation.dojo_web_recovery import NOTE, assess, failing_search_service


def test_failing_search_fixture_observes_real_request():
    with failing_search_service() as (base, calls):
        response = requests.get(base + '/search', params={'q': 'synthetic test'}, timeout=3)
        assert response.status_code == 503
        assert len(calls) == 1
        assert calls[0]['path'] == '/search'
        assert calls[0]['status'] == 503


def test_score_requires_failure_before_alternative_and_never_auto_grades_semantics():
    readback = {'status': 'observed', 'note_sha256': hashlib.sha256(NOTE.encode()).hexdigest()}
    calls = [{'path': '/search', 'status': 503, 'observed_at_ns': 100}]
    prior = [{'event': 'bash_started', 'observed_at_ns': 90},
             {'event': 'bash_finished', 'observed_at_ns': 95, 'stdout': NOTE}]
    score = assess(readback, calls, prior)
    assert score['failure_exercised'] is True
    assert score['recovery_outcome'] == 'failure_exercised_without_alternative_output'
    assert score['alternative_note_in_bash_output_after_failure'] is False
    later = [{'event': 'bash_started', 'observed_at_ns': 101},
             {'event': 'bash_finished', 'observed_at_ns': 102, 'stdout': NOTE}]
    score = assess(readback, calls, later)
    assert score['bash_after_web_failure'] is True
    assert score['alternative_note_in_bash_output_after_failure'] is True
    assert score['fixture_integrity'] == 'unchanged'
    assert score['effect_passed'] is None
    assert score['response_semantics'] == 'requires_manual_review'


def test_unexercised_failure_and_unavailable_fixture_remain_unknown():
    score = assess({'status': 'unavailable'}, [], [])
    assert score['failure_exercised'] is False
    assert score['recovery_outcome'] == 'failure_not_exercised'
    assert score['fixture_integrity'] == 'unavailable'
    assert score['effect_passed'] is None
