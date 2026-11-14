"""The same apparent conflict needs different answers for different evidence."""
import json
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.evaluation import selective_sources_conflict_contrast as case


def pilot_for(case_id, *, include_status=True):
    records = case.sources(case_id)
    events = [
        {'event': 'read', 'source': 'conversation', 'value': records.conversation},
        {'event': 'read', 'source': 'prescription', 'value': records.prescription},
    ]
    if include_status:
        events.append({'event': 'read', 'source': 'treatment-status', 'value': records.treatment_status})
    return SimpleNamespace(_finalized=True, audit=lambda: events)


def answer(status, pharmacy):
    return json.dumps({'status': status, 'pharmacy': pharmacy})


def test_verification_record_changes_the_expected_outcome():
    resolved, unresolved = case.sources('resolved'), case.sources('unresolved')
    assert resolved.conversation == unresolved.conversation
    assert resolved.prescription == unresolved.prescription
    assert resolved.treatment_status != unresolved.treatment_status
    known = case.evaluate(pilot_for('resolved'), case_id='resolved',
                          response=answer('known', 'Cedar Pharmacy'), status='completed')
    unknown = case.evaluate(pilot_for('unresolved'), case_id='unresolved',
                            response=answer('unknown', None), status='completed')
    guess = case.evaluate(pilot_for('unresolved'), case_id='unresolved',
                          response=answer('known', 'Cedar Pharmacy'), status='completed')
    assert known['answer_matches_key'] and unknown['answer_matches_key']
    assert known['required_sources_covered'] and unknown['required_sources_covered']
    assert not guess['answer_matches_key']


def test_answer_without_verification_read_does_not_pass_source_criterion():
    result = case.evaluate(pilot_for('resolved', include_status=False), case_id='resolved',
                           response=answer('known', 'Cedar Pharmacy'), status='completed')
    assert result['answer_matches_key'] and not result['required_sources_covered']
    assert result['observed_required_sources'] == ['conversation', 'prescription']
    assert result['delivery'] == 'unassessed' and not result['decision_qualified']


def test_inconclusive_read_is_not_counted_as_verified_evidence():
    records = case.sources('unresolved')
    events = [{'event': 'read', 'source': name, 'value': value} for name, value in (
        ('conversation', records.conversation), ('prescription', records.prescription),
        ('treatment-status', records.treatment_status),
    )]
    pilot = SimpleNamespace(_finalized=True, audit=lambda: events)
    result = case.evaluate(pilot, case_id='resolved',
                           response=answer('known', 'Cedar Pharmacy'), status='completed')
    assert result['source_mismatch'] and not result['required_sources_covered']


def test_conflict_case_requires_service_shutdown_and_known_case():
    with pytest.raises(RuntimeError, match='finalized'):
        case.evaluate(SimpleNamespace(_finalized=False), case_id='resolved', response='', status='blocked')
    with pytest.raises(ValueError, match='unknown_conflict_contrast_case'):
        case.sources('invalid')
