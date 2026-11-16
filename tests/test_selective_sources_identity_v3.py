"""Privacy and interpretation of the public development-case diagnostic."""
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.evaluation import selective_sources_identity_v3 as diagnostic
from alphonse.agent_v2.evaluation.selective_sources_identity_v3 import answer_shape, assess_question


@pytest.mark.parametrize(('response', 'expected'), [
    ('{"status":"known","pharmacy":"synthetic-private-value"}',
     {'format': 'object', 'status': 'known', 'pharmacy': 'string'}),
    ('{"status":"unknown","pharmacy":null}',
     {'format': 'object', 'status': 'unknown', 'pharmacy': 'null'}),
    ('{"status":"synthetic-private-value","pharmacy":23}',
     {'format': 'object', 'status': 'other', 'pharmacy': 'other_type'}),
    ('synthetic-private-value', {'format': 'invalid_json'}),
    ('', {'format': 'no_answer'}),
])
def test_answer_shape_is_categorical_and_does_not_retain_content(response, expected):
    result = answer_shape(response)
    assert result == expected
    assert 'synthetic-private-value' not in repr(result)


@pytest.mark.parametrize(('status', 'response', 'assessor', 'expected'), [
    ('completed', 'synthetic answer', lambda *_: 'useful', ('not_asked', 'candidate_status')),
    ('waiting_user', '', lambda *_: 'useful', ('unassessed', 'missing_question')),
    ('waiting_user', 'private question?', None, ('unassessed', 'no_assessor')),
    ('waiting_user', 'private question?', lambda *_: 'useful', ('useful', 'caller_assessor_unverified')),
    ('waiting_user', 'private question?', lambda *_: 'unexpected private text', ('unassessed', 'invalid_assessment')),
])
def test_question_assessment_is_bounded_and_does_not_retain_text(status, response, assessor, expected):
    result = assess_question('missing', status, response, assessor)
    assert (result['status'], result['basis']) == expected
    assert 'private' not in repr(result)


def test_question_assessor_runs_only_after_source_service_shutdown(monkeypatch):
    class FakePilot:
        def __init__(self, _sources):
            self._finalized = False
        def __enter__(self):
            return self
        def __exit__(self, *_):
            self._finalized = True
        @property
        def url(self):
            return 'synthetic-url'
        def audit(self):
            return []
    pilot = FakePilot(None)
    monkeypatch.setattr(diagnostic, 'LoopbackPrescriptionPilot', lambda _: pilot)
    calls = []
    def assessor(case_id, text, observed_sources):
        assert pilot._finalized
        calls.append(case_id)
        assert text == 'Which record should I use?'
        assert observed_sources == ()
        return 'useful'
    result = diagnostic.run('missing', lambda _: SimpleNamespace(status='waiting_user',
        response='Which record should I use?'), question_assessor=assessor)
    assert calls == ['missing']
    assert result['question_assessment'] == {'status': 'useful', 'basis': 'caller_assessor_unverified'}
    assert result['question_content'] == 'assessed_by_caller'
    assert result['delivery'] == 'unassessed'
    assert 'Which record should I use?' not in repr(result)


def test_question_assessor_error_is_not_exposed():
    def broken(*_):
        raise RuntimeError('private assessor exception')
    assert assess_question('missing', 'waiting_user', 'private question?', broken) == {
        'status': 'unassessed', 'basis': 'assessor_failed'}
