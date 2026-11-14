"""Verify that a source-identity contrast changes the independent answer."""
import json
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.evaluation import selective_sources_reference_contrast as case


def pilot_for(case_id, *, include_conversation=True):
    records = case.sources(case_id)
    events = [{'event': 'read', 'source': 'prescription', 'value': records.prescription}]
    if include_conversation:
        events.append({'event': 'read', 'source': 'conversation', 'value': records.conversation})
    return SimpleNamespace(_finalized=True, audit=lambda: events)


def answer(name):
    return json.dumps({'status': 'known', 'pharmacy': name})


def test_only_the_current_reference_changes_the_correct_answer():
    first, second = case.sources('first_current'), case.sources('second_current')
    assert first.prescription == second.prescription
    assert first.treatment_status == second.treatment_status
    assert first.conversation != second.conversation
    a = case.evaluate(pilot_for('first_current'), case_id='first_current',
                      response=answer('Cedar Pharmacy'), status='completed')
    b = case.evaluate(pilot_for('second_current'), case_id='second_current',
                      response=answer('Maple Pharmacy'), status='completed')
    wrong = case.evaluate(pilot_for('second_current'), case_id='second_current',
                          response=answer('Cedar Pharmacy'), status='completed')
    assert a['answer_matches_key'] and b['answer_matches_key'] and not wrong['answer_matches_key']
    assert a['required_sources_covered'] and b['required_sources_covered']


def test_correct_guess_without_conversation_is_reported_separately():
    result = case.evaluate(pilot_for('first_current', include_conversation=False),
                           case_id='first_current', response=answer('Cedar Pharmacy'), status='completed')
    assert result['answer_matches_key'] and not result['required_sources_covered']
    assert result['observed_required_sources'] == ['prescription']
    assert result['response_order'] == 'unassessed' and result['delivery'] == 'unassessed'


def test_evaluator_requires_shutdown_and_known_case():
    with pytest.raises(RuntimeError, match='finalized'):
        case.evaluate(SimpleNamespace(_finalized=False), case_id='first_current', response='', status='blocked')
    with pytest.raises(ValueError, match='unknown_reference_contrast_case'):
        case.sources('not-a-case')
