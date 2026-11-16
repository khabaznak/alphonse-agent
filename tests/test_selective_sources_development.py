from types import SimpleNamespace

import pytest

from alphonse.agent_v2.evaluation import selective_sources_development as case


class FakePilot:
    def __init__(self, events, finalized=True):
        self._events = events
        self._finalized = finalized

    def audit(self):
        return self._events


def test_reading_required_sources_without_distractor():
    data = case.sources()
    pilot = FakePilot([
        {'event': 'read', 'source': 'index', 'value': {}},
        {'event': 'read', 'source': 'conversation', 'value': data.conversation},
        {'event': 'read', 'source': 'prescription', 'value': data.prescription},
    ])
    result = case.evaluate(pilot, response='Cedar Pharmacy', status='completed')
    assert result['answer_matches_key']
    assert result['required_sources_covered']
    assert result['index_read']
    assert not result['distractor_read']
    assert result['unauthorized_access_attempts'] == 'unassessed'


def test_distractor_and_missing_sources_are_independent_of_answer():
    pilot = FakePilot([{'event': 'read', 'source': 'treatment-status', 'value': {}}])
    result = case.evaluate(pilot, response='Cedar Pharmacy', status='completed')
    assert result['answer_matches_key']
    assert not result['required_sources_covered']
    assert result['distractor_read']


def test_wrong_answer_and_blocker_are_reported_without_claiming_success():
    result = case.evaluate(FakePilot([]), response='Maple Pharmacy', status='blocked')
    assert not result['answer_matches_key']
    assert not result['required_sources_covered']
    assert result['candidate_status'] == 'blocked'


def test_key_only_evaluated_after_shutdown():
    with pytest.raises(RuntimeError, match='pilot_must_be_finalized'):
        case.evaluate(FakePilot([], finalized=False), response='Cedar Pharmacy', status='completed')


def test_runner_shuts_down_before_scoring_and_only_passes_url(monkeypatch):
    class FakeService(FakePilot):
        def __init__(self, data):
            super().__init__([], finalized=False)
            self.data = data
            self.url = 'http://fixture/'

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self._finalized = True

    monkeypatch.setattr(case, 'LoopbackPrescriptionPilot', FakeService)
    result = case.run(lambda url: SimpleNamespace(response=url, status='completed'))
    assert not result['answer_matches_key']
    assert not result['required_sources_covered']
