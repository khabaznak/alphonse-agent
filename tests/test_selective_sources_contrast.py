"""The same source can be irrelevant to one request and necessary for another."""
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.evaluation import selective_sources_contrast as contrast
from alphonse.agent_v2.evaluation import selective_sources_development as pharmacy


class Pilot:
    def __init__(self, events, finalized=True):
        self.events = events
        self._finalized = finalized

    def audit(self):
        return self.events


def test_same_status_read_is_distractor_for_pharmacy_but_required_for_exercise():
    event = {'event': 'read', 'source': 'treatment-status',
             'value': {'exercise_program': 'stretching', 'status': 'paused'}}
    pharmacy_result = pharmacy.evaluate(Pilot([event]), response='Cedar Pharmacy', status='completed')
    exercise_result = contrast.evaluate(Pilot([event]), response='stretching', status='completed')
    assert pharmacy_result['distractor_read'] and not pharmacy_result['required_sources_covered']
    assert exercise_result['required_source_covered'] and exercise_result['answer_matches_key']
    assert exercise_result['irrelevant_sources_read'] == []


def test_answer_does_not_replace_valid_source_or_erase_unnecessary_reads():
    pilot = Pilot([
        {'event': 'read', 'source': 'treatment-status', 'value': {'exercise_program': 'stretching', 'status': 'active'}},
        {'event': 'read', 'source': 'prescription', 'value': {}},
    ])
    result = contrast.evaluate(pilot, response='stretching', status='completed')
    assert result['answer_matches_key'] and not result['required_source_covered']
    assert result['source_mismatch'] and result['irrelevant_sources_read'] == ['prescription']
    assert result['response_order'] == result['delivery'] == 'unassessed'


def test_scoring_waits_for_shutdown_and_runner_passes_only_url(monkeypatch):
    with pytest.raises(RuntimeError, match='pilot_must_be_finalized'):
        contrast.evaluate(Pilot([], finalized=False), response='stretching', status='completed')

    class FakeService(Pilot):
        def __init__(self, _data):
            super().__init__([], finalized=False)
            self.url = 'http://synthetic/'

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self._finalized = True

    monkeypatch.setattr(contrast, 'LoopbackPrescriptionPilot', FakeService)
    result = contrast.run(lambda url: SimpleNamespace(response=url, status='completed'))
    assert not result['answer_matches_key'] and not result['required_source_covered']
