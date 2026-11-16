"""The gym must distinguish an evidence-backed unknown from a guess."""
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.evaluation import selective_sources_ambiguity as case


class Pilot:
    def __init__(self, events=(), finalized=True):
        self.events = events
        self._finalized = finalized

    def audit(self):
        return list(self.events)


def read(source, value):
    return {'event': 'read', 'source': source, 'value': value}


def test_conflicting_records_require_unknown_and_both_matching_sources():
    events = [read('conversation', {'current_prescription_id': 'rx-41'}),
              read('prescription', {'prescriptions': [
                  {'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy'},
                  {'id': 'rx-41', 'pharmacy': 'Maple Pharmacy'}]})]
    result = case.evaluate(Pilot(events), response='{"status":"unknown","pharmacy":null}', status='completed')
    assert result['answer_matches_key'] and result['required_sources_covered']
    assert not result['unrelated_source_read'] and not result['source_mismatch']
    for answer in ('{"status":"known","pharmacy":"Cedar Pharmacy"}',
                   '{"status":"unknown","pharmacy":""}', 'unknown'):
        assert not case.evaluate(Pilot(events), response=answer, status='completed')['answer_matches_key']
    assert not case.evaluate(Pilot(events[:1]), response='{"status":"unknown","pharmacy":null}',
                             status='completed')['required_sources_covered']


def test_wrong_source_content_or_question_does_not_pass_as_verified_unknown():
    events = [read('conversation', {'current_prescription_id': 'rx-41'}),
              read('prescription', {'prescriptions': []}),
              read('treatment-status', {'exercise_program': 'stretching', 'status': 'paused'})]
    result = case.evaluate(Pilot(events), response='Which pharmacy?', status='waiting_user')
    assert not result['answer_matches_key'] and not result['required_sources_covered']
    assert result['source_mismatch'] and result['unrelated_source_read']
    assert result['candidate_status'] == 'waiting_user'
    assert result['question_content'] == result['delivery'] == 'unassessed'


def test_scoring_requires_shutdown_and_does_not_expose_key_to_callback(monkeypatch):
    with pytest.raises(RuntimeError, match='pilot_must_be_finalized'):
        case.evaluate(Pilot(finalized=False), response='', status='blocked')

    class FakeService(Pilot):
        def __init__(self, _data):
            super().__init__(finalized=False)
            self.url = 'http://synthetic/'
        def __enter__(self):
            return self
        def __exit__(self, *_):
            self._finalized = True

    monkeypatch.setattr(case, 'LoopbackPrescriptionPilot', FakeService)
    result = case.run(lambda url: SimpleNamespace(response=url, status='completed'))
    assert not result['answer_matches_key'] and not result['required_sources_covered']
