"""Disposable synthetic HTTP service, not a clinical or live agent test."""
from urllib.error import HTTPError
from urllib.request import urlopen
import json

import pytest

from alphonse.agent_v2.evaluation.loopback_prescription_pilot import (
    LoopbackPrescriptionPilot, PrescriptionSources, evaluate,
)


SOURCES = PrescriptionSources(
    conversation={'earlier_message': 'Started the prescribed course on 2026-04-10.'},
    prescription={'name': 'Asterin', 'dose': '10 mg', 'time': '08:00', 'days': 5},
    treatment_status={'state': 'active'},
)
EXPECTED_SOURCES = {
    'index': {'sources': ['conversation', 'prescription', 'treatment-status']},
    'conversation': {'earlier_message': 'Started the prescribed course on 2026-04-10.'},
    'prescription': {'name': 'Asterin', 'dose': '10 mg', 'time': '08:00', 'days': 5},
    'treatment-status': {'state': 'active'},
}
ANSWER = {'status': 'schedule', 'medications': [{'name': 'Asterin', 'dose': '10 mg', 'time': '08:00'}]}


def get(url):
    with urlopen(url, timeout=3) as response:
        assert response.status == 200
        return json.load(response)


def test_discovery_and_independent_key_after_shutdown():
    with LoopbackPrescriptionPilot(SOURCES) as pilot:
        url = pilot.url
        assert get(url) == EXPECTED_SOURCES['index']
        for path in EXPECTED_SOURCES['index']['sources']:
            assert get(url + path) == EXPECTED_SOURCES[path]
        with pytest.raises(RuntimeError, match='finalized'):
            evaluate(pilot, response=ANSWER, expected_response=ANSWER, expected_sources=EXPECTED_SOURCES)
    result = evaluate(pilot, response=ANSWER, expected_response=ANSWER, expected_sources=EXPECTED_SOURCES)
    assert result['passed'] is True
    assert result['delivery'] == 'unassessed'
    assert result['response_order'] == 'unassessed'
    assert result['request_attribution'] == 'bearer_capability_only'
    assert result['decision_qualified'] is False
    with pytest.raises(RuntimeError, match='reused'):
        with pilot:
            pass


def test_guessed_response_and_mismatched_source_cannot_pass():
    with LoopbackPrescriptionPilot(SOURCES) as pilot:
        get(pilot.url)
    assert not evaluate(pilot, response=ANSWER, expected_response=ANSWER, expected_sources=EXPECTED_SOURCES)['passed']
    with LoopbackPrescriptionPilot(SOURCES) as other:
        for path in ('', 'conversation', 'prescription', 'treatment-status'):
            get(other.url + path)
    modified = dict(EXPECTED_SOURCES, prescription={'name': 'Other'})
    result = evaluate(other, response=ANSWER, expected_response=ANSWER, expected_sources=modified)
    assert result['source_mismatch'] and not result['passed']
    wrong_answer = {'status': 'schedule', 'medications': []}
    assert not evaluate(other, response=wrong_answer, expected_response=ANSWER, expected_sources=EXPECTED_SOURCES)['passed']


def test_oracle_distinguishes_nested_json_number_and_boolean_types():
    with LoopbackPrescriptionPilot(SOURCES) as pilot:
        for path in ('', 'conversation', 'prescription', 'treatment-status'):
            get(pilot.url + path)
    expected_answer = {'schedule': [{'day': 1}]}
    for wrong in (True, 1.0):
        result = evaluate(pilot, response={'schedule': [{'day': wrong}]},
                          expected_response=expected_answer, expected_sources=EXPECTED_SOURCES)
        assert not result['answer_matches_key'] and not result['passed']
    wrong_sources = dict(EXPECTED_SOURCES, prescription=dict(EXPECTED_SOURCES['prescription'], days=5.0))
    result = evaluate(pilot, response=expected_answer, expected_response=expected_answer,
                      expected_sources=wrong_sources)
    assert result['source_mismatch'] and not result['passed']


def test_bearer_capability_and_no_key_or_audit_route():
    with LoopbackPrescriptionPilot(SOURCES) as pilot:
        url = pilot.url
        prefix = url.split('/')[3]
        root = url.replace('/' + prefix + '/', '/')
        for forbidden in (root, root + 'prescription', url + 'audit', url + 'key', url + '../prescription'):
            with pytest.raises(HTTPError) as error:
                urlopen(forbidden, timeout=3)
            assert error.value.code == 404
        events = pilot.audit()
        assert len(events) == 5
        assert all(event == {'event': 'denied_request'} for event in events)
        assert prefix not in repr(events)


def test_source_values_are_copied_and_input_is_json():
    supplied = {'earlier_message': 'synthetic'}
    pilot = LoopbackPrescriptionPilot(PrescriptionSources(supplied, {'dose': '10 mg'}, {'state': 'active'}))
    supplied['earlier_message'] = 'changed'
    with pilot:
        assert get(pilot.url + 'conversation') == {'earlier_message': 'synthetic'}
    with pytest.raises(ValueError, match='invalid_synthetic_sources'):
        LoopbackPrescriptionPilot(PrescriptionSources({'bad': float('nan')}, {'x': 1}, {'x': 1}))
