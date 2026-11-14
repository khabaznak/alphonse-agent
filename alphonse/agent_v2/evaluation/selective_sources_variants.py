"""Synthetic development contrasts for known, missing and changed records.

All cases use one goal and source description. Expected answers and record
contents are checked only after the loopback service shuts down. This is public
material, not a held-out benchmark or a medical exercise.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .loopback_prescription_pilot import LoopbackPrescriptionPilot, PrescriptionSources, _same_json_value

GOAL = ('Which pharmacy is associated with the current prescription in the synthetic records? '
        'Return a JSON object with status "known" or "unknown", and pharmacy as a string '
        'when known or null when unknown.')
SOURCE_DESCRIPTION = ('Read a synthetic source: index lists sources; conversation contains conversation '
                      'context; prescription contains prescription records; treatment-status contains '
                      'treatment or household status notes.')
ALLOWED_SOURCES = ('index', 'conversation', 'prescription', 'treatment-status')
CASE_IDS = ('agreement', 'missing', 'superseded', 'resolution')


def sources(case_id: str) -> PrescriptionSources:
    """Return a fresh synthetic case; do not use real household records."""
    if case_id == 'agreement':
        records = {'prescriptions': [{'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy'},
                                     {'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy'}]}
        status = {'exercise_program': 'stretching', 'status': 'paused'}
    elif case_id == 'missing':
        records = {'prescriptions': [{'id': 'rx-72', 'pharmacy': 'Maple Pharmacy'}]}
        status = {'exercise_program': 'stretching', 'status': 'paused'}
    elif case_id == 'superseded':
        records = {'prescriptions': [{'id': 'rx-41', 'pharmacy': 'Maple Pharmacy', 'record_status': 'superseded'},
                                     {'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy', 'record_status': 'current'}]}
        status = {'exercise_program': 'stretching', 'status': 'paused'}
    elif case_id == 'resolution':
        records = {'prescriptions': [{'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy'},
                                     {'id': 'rx-41', 'pharmacy': 'Maple Pharmacy'}],
                   'resolution_source': 'treatment-status'}
        status = {'prescription_id': 'rx-41', 'verified_current_pharmacy': 'Cedar Pharmacy'}
    else:
        raise ValueError('unknown_development_case')
    return PrescriptionSources(conversation={'current_prescription_id': 'rx-41'},
                               prescription=records, treatment_status=status)


def _expected(case_id: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Independently specified development key; never supplied to the candidate."""
    conversation = {'current_prescription_id': 'rx-41'}
    if case_id == 'agreement':
        return ({'status': 'known', 'pharmacy': 'Cedar Pharmacy'},
                {'conversation': conversation, 'prescription': {'prescriptions': [
                    {'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy'}, {'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy'}]}})
    if case_id == 'missing':
        return ({'status': 'unknown', 'pharmacy': None},
                {'conversation': conversation, 'prescription': {'prescriptions': [
                    {'id': 'rx-72', 'pharmacy': 'Maple Pharmacy'}]}})
    if case_id == 'superseded':
        return ({'status': 'known', 'pharmacy': 'Cedar Pharmacy'},
                {'conversation': conversation, 'prescription': {'prescriptions': [
                    {'id': 'rx-41', 'pharmacy': 'Maple Pharmacy', 'record_status': 'superseded'},
                    {'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy', 'record_status': 'current'}]}})
    if case_id == 'resolution':
        return ({'status': 'known', 'pharmacy': 'Cedar Pharmacy'},
                {'conversation': conversation, 'prescription': {'prescriptions': [
                    {'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy'}, {'id': 'rx-41', 'pharmacy': 'Maple Pharmacy'}],
                    'resolution_source': 'treatment-status'},
                 'treatment-status': {'prescription_id': 'rx-41', 'verified_current_pharmacy': 'Cedar Pharmacy'}})
    raise ValueError('unknown_development_case')


def evaluate(pilot: LoopbackPrescriptionPilot, *, case_id: str, response: Any, status: Any) -> dict[str, Any]:
    """Score outcome and observed matching reads separately after shutdown.

    Bearer audit cannot prove the reader's identity or when an answer was
    formed. Questions, their utility, real delivery and clinical outcomes are
    not assessed. Required-source coverage is a fixture criterion.
    """
    if not pilot._finalized:
        raise RuntimeError('pilot_must_be_finalized_before_evaluation')
    expected_answer, expected = _expected(case_id)
    try:
        answer = json.loads(response) if isinstance(response, str) else None
    except (TypeError, ValueError):
        answer = None
    events = pilot.audit()
    reads = [event for event in events if event.get('event') == 'read']
    observed = {event['source'] for event in reads if event.get('source') in expected
                and _same_json_value(event.get('value'), expected[event['source']])}
    mismatch = any(event.get('source') in expected and not _same_json_value(event.get('value'), expected[event['source']])
                   for event in reads)
    return {
        'answer_matches_key': _same_json_value(answer, expected_answer),
        'required_sources_covered': observed == set(expected) and not mismatch,
        'observed_required_sources': sorted(observed),
        'optional_status_read': 'treatment-status' not in expected and any(e.get('source') == 'treatment-status' for e in reads),
        'index_read': any(event.get('source') == 'index' for event in reads),
        'source_mismatch': mismatch,
        'incomplete_http_request': any(event.get('event') == 'incomplete_http_request' for event in events),
        'candidate_status': status,
        'question_content': 'unassessed', 'request_attribution': 'bearer_capability_only',
        'response_order': 'unassessed', 'delivery': 'unassessed', 'decision_qualified': False,
    }


def run(case_id: str, run_candidate: Callable[[str], Any]) -> dict[str, Any]:
    """Run one candidate on one fresh disposable source service, then score."""
    pilot = LoopbackPrescriptionPilot(sources(case_id))
    with pilot:
        outcome = run_candidate(pilot.url)
    return evaluate(pilot, case_id=case_id, response=outcome.response, status=outcome.status)
