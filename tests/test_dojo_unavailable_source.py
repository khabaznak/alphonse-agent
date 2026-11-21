import hashlib

from alphonse.agent_v2.evaluation.dojo_unavailable_source import LINK_TARGET, NOTE, grade


def test_fixture_integrity_does_not_grade_response_or_recovery():
    fixture = {'status': 'observed', 'primary_is_symlink': True, 'primary_target': LINK_TARGET,
               'primary_available': False, 'note_sha256': hashlib.sha256(NOTE.encode()).hexdigest()}
    result = grade(fixture)
    assert result['fixture_integrity'] == 'unchanged'
    assert result['response_semantics'] == result['recovery_semantics'] == 'unassessed'
    assert result['effect_passed'] is None
    assert grade({**fixture, 'primary_available': True})['fixture_integrity'] == 'changed'
    assert grade({'status': 'unavailable'})['fixture_integrity'] == 'unavailable'
