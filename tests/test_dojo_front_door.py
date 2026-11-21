import hashlib

from alphonse.agent_v2.evaluation.dojo_front_door import RECORD_TEXT, grade


def test_front_door_fixture_grading_does_not_claim_agent_success():
    result = grade({'status': 'observed', 'record_sha256': hashlib.sha256(RECORD_TEXT.encode()).hexdigest()})
    assert result['fixture_integrity'] == 'unchanged'
    assert result['effect_passed'] is None and result['response_semantics'] == 'unassessed'
    assert grade({'status': 'observed', 'record_sha256': 'wrong'})['fixture_integrity'] == 'changed'
    assert grade({'status': 'unavailable'})['fixture_integrity'] == 'unavailable'
