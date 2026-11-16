"""A current reference changes whether conflicting records are resolvable."""
import json
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.evaluation import selective_sources_identity_conflict as case


def pilot_for(case_id, *, missing=None, replacement=None):
    records = case.sources(case_id)
    values = {'conversation': records.conversation, 'prescription': records.prescription,
              'treatment-status': records.treatment_status}
    if replacement is not None:
        values['conversation'] = replacement
    events = [{'event': 'read', 'source': name, 'value': value}
              for name, value in values.items() if name != missing]
    return SimpleNamespace(_finalized=True, audit=lambda: events)


def answer(status, pharmacy):
    return json.dumps({'status': status, 'pharmacy': pharmacy})


def test_current_reference_changes_answer_while_other_records_are_identical():
    first, second = (case.sources(name) for name in case.CASE_IDS)
    assert first.conversation != second.conversation
    assert first.prescription == second.prescription
    assert first.treatment_status == second.treatment_status
    expected = [('current_verified', answer('known', 'Cedar Pharmacy')),
                ('current_inconclusive', answer('unknown', None))]
    for case_id, response in expected:
        result = case.evaluate(pilot_for(case_id), case_id=case_id, response=response, status='completed')
        assert result['answer_matches_key'] and result['required_sources_covered']
    guessed = case.evaluate(pilot_for('current_inconclusive'), case_id='current_inconclusive',
                            response=answer('known', 'Cedar Pharmacy'), status='completed')
    assert not guessed['answer_matches_key']


@pytest.mark.parametrize('missing', ['conversation', 'prescription', 'treatment-status'])
def test_expected_answer_alone_does_not_supply_missing_observation(missing):
    result = case.evaluate(pilot_for('current_verified', missing=missing), case_id='current_verified',
                           response=answer('known', 'Cedar Pharmacy'), status='completed')
    assert result['answer_matches_key'] and not result['required_sources_covered']
    assert result['delivery'] == 'unassessed' and not result['decision_qualified']


def test_changed_reference_and_unfinalized_service_are_rejected():
    wrong = case.sources('current_inconclusive').conversation
    result = case.evaluate(pilot_for('current_verified', replacement=wrong), case_id='current_verified',
                           response=answer('known', 'Cedar Pharmacy'), status='completed')
    assert result['source_mismatch'] and not result['required_sources_covered']
    with pytest.raises(RuntimeError, match='finalized'):
        case.evaluate(SimpleNamespace(_finalized=False), case_id='current_verified', response='', status='blocked')
