"""Contrasts ensure 'always unknown' and 'stop at conflict' cannot pass."""
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.evaluation import selective_sources_variants as cases


class Pilot:
    def __init__(self, events=(), finalized=True):
        self.events = events
        self._finalized = finalized

    def audit(self):
        return list(self.events)


def read(name, content):
    return {'event': 'read', 'source': name, 'value': content}


@pytest.mark.parametrize(('case_id', 'prescription', 'response'), [
    ('agreement', {'prescriptions': [{'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy'},
                                    {'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy'}]},
     '{"status":"known","pharmacy":"Cedar Pharmacy"}'),
    ('missing', {'prescriptions': [{'id': 'rx-72', 'pharmacy': 'Maple Pharmacy'}]},
     '{"status":"unknown","pharmacy":null}'),
    ('superseded', {'prescriptions': [{'id': 'rx-41', 'pharmacy': 'Maple Pharmacy', 'record_status': 'superseded'},
                                     {'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy', 'record_status': 'current'}]},
     '{"status":"known","pharmacy":"Cedar Pharmacy"}'),
])
def test_same_structured_answer_can_be_known_or_unknown_based_on_records(case_id, prescription, response):
    pilot = Pilot([read('conversation', {'current_prescription_id': 'rx-41'}), read('prescription', prescription)])
    result = cases.evaluate(pilot, case_id=case_id, response=response, status='completed')
    assert result['answer_matches_key'] and result['required_sources_covered']
    wrong = '{"status":"known","pharmacy":"Maple Pharmacy"}' if case_id == 'missing' else '{"status":"unknown","pharmacy":null}'
    assert not cases.evaluate(pilot, case_id=case_id, response=wrong, status='completed')['answer_matches_key']


def test_resolving_conflict_requires_observed_matching_third_source():
    common = [read('conversation', {'current_prescription_id': 'rx-41'}),
              read('prescription', {'prescriptions': [{'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy'},
                                                      {'id': 'rx-41', 'pharmacy': 'Maple Pharmacy'}],
                                    'resolution_source': 'treatment-status'})]
    answer = '{"status":"known","pharmacy":"Cedar Pharmacy"}'
    before = cases.evaluate(Pilot(common), case_id='resolution', response=answer, status='completed')
    assert before['answer_matches_key'] and not before['required_sources_covered']
    verified = common + [read('treatment-status', {'prescription_id': 'rx-41',
                                                   'verified_current_pharmacy': 'Cedar Pharmacy'})]
    after = cases.evaluate(Pilot(verified), case_id='resolution', response=answer, status='completed')
    assert after['required_sources_covered'] and not after['optional_status_read']
    bad = common + [read('treatment-status', {'prescription_id': 'rx-41',
                                              'verified_current_pharmacy': 'Maple Pharmacy'})]
    mismatch = cases.evaluate(Pilot(bad), case_id='resolution', response=answer, status='completed')
    assert mismatch['source_mismatch'] and not mismatch['required_sources_covered']


def test_unrelated_reads_are_measured_separately_and_nonanswers_not_graded():
    events = [read('conversation', {'current_prescription_id': 'rx-41'}),
              read('prescription', {'prescriptions': [{'id': 'rx-72', 'pharmacy': 'Maple Pharmacy'}]}),
              read('treatment-status', {'exercise_program': 'stretching', 'status': 'paused'}),
              read('index', {'sources': []})]
    result = cases.evaluate(Pilot(events), case_id='missing', response='Which pharmacy?', status='waiting_user')
    assert result['optional_status_read'] and result['index_read'] and result['required_sources_covered']
    assert not result['answer_matches_key']
    assert result['question_content'] == result['delivery'] == 'unassessed'


def test_shutdown_and_unknown_case_are_enforced_without_exposing_oracle(monkeypatch):
    with pytest.raises(RuntimeError, match='pilot_must_be_finalized'):
        cases.evaluate(Pilot(finalized=False), case_id='missing', response='', status='blocked')
    with pytest.raises(ValueError, match='unknown_development_case'):
        cases.sources('not-a-case')

    class Service(Pilot):
        def __init__(self, _sources):
            super().__init__(finalized=False)
            self.url = 'http://synthetic/'
        def __enter__(self):
            return self
        def __exit__(self, *_):
            self._finalized = True

    monkeypatch.setattr(cases, 'LoopbackPrescriptionPilot', Service)
    seen = []
    result = cases.run('resolution', lambda url: seen.append(url) or SimpleNamespace(response='', status='blocked'))
    assert seen == ['http://synthetic/']
    assert not result['answer_matches_key'] and not result['required_sources_covered']
