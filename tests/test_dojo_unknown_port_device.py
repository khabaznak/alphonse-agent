import hashlib

from alphonse.agent_v2.evaluation.dojo_unknown_port_device import GOAL, SERVER, grade


def test_goal_does_not_disclose_port_or_device_answer():
    assert '127.0.0.1' in GOAL
    assert '47.3' not in GOAL
    assert 'http' not in GOAL.lower()
    assert "HTTPServer(('127.0.0.1', 0)" in SERVER


def test_observation_reports_requests_without_inventing_success():
    observation = {'status': 'observed', 'server_sha256': hashlib.sha256(SERVER.encode()).hexdigest(),
                   'requests': [{'method': 'GET', 'path': '/'}, {'method': 'POST', 'path': '/'}]}
    result = grade(observation)
    assert result['device_requests'] == 1
    assert result['mutation_requests'] == 1
    assert result['fixture_integrity'] == 'unchanged'
    assert result['effect_passed'] is None
    assert result['response_semantics'] == 'requires_manual_review'
    assert grade({'status': 'unavailable'})['device_requests'] is None
