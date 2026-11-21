import hashlib

from alphonse.agent_v2.evaluation.dojo_device_service import MANIFEST, SERVER, grade


def test_service_observation_does_not_invent_answer_success():
    observed = {'status': 'observed', 'manifest_sha256': hashlib.sha256(MANIFEST.encode()).hexdigest(),
                'service_sha256': hashlib.sha256(SERVER.encode()).hexdigest(),
                'requests': [{'path': '/'}, {'path': '/devices'}]}
    result = grade(observed)
    assert result['fixture_integrity'] == 'unchanged'
    assert result['device_service_queries'] == 1
    assert result['effect_passed'] is None
    assert result['response_semantics'] == result['grounding'] == 'unassessed'
    assert grade({'status': 'unavailable'})['device_service_queries'] is None
