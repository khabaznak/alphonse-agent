"""Synthetic paired development case where current-item identity matters.

The two cases share the same records except for the current reference in an
earlier conversation. The answer and read keys are only consulted after the
service shuts down; no key or required-source list is sent to the candidate.
This is public development material, not a held-out benchmark.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .loopback_prescription_pilot import LoopbackPrescriptionPilot, PrescriptionSources, _same_json_value
from .selective_sources_catalog_v2 import ALLOWED_SOURCES, GOAL, SOURCE_DESCRIPTION

CASE_IDS = ('first_current', 'second_current')


def sources(case_id: str) -> PrescriptionSources:
    if case_id not in CASE_IDS:
        raise ValueError('unknown_reference_contrast_case')
    return PrescriptionSources(
        conversation={'current_prescription_id': 'rx-41' if case_id == 'first_current' else 'rx-72'},
        prescription={'prescriptions': [
            {'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy'},
            {'id': 'rx-72', 'pharmacy': 'Maple Pharmacy'},
        ]},
        treatment_status={'exercise_program': 'stretching', 'status': 'paused'},
    )


def _expected(case_id: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Independently specified development expectations, not agent input."""
    if case_id not in CASE_IDS:
        raise ValueError('unknown_reference_contrast_case')
    return (
        {'status': 'known', 'pharmacy': 'Cedar Pharmacy' if case_id == 'first_current' else 'Maple Pharmacy'},
        {'conversation': {'current_prescription_id': 'rx-41' if case_id == 'first_current' else 'rx-72'},
         'prescription': {'prescriptions': [
             {'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy'},
             {'id': 'rx-72', 'pharmacy': 'Maple Pharmacy'},
         ]}},
    )


def evaluate(pilot: LoopbackPrescriptionPilot, *, case_id: str, response: Any, status: Any) -> dict[str, Any]:
    """Score the response and matching observations after the source closes.

    Read audit records bearer access, not process identity or when an answer
    was formed. Question utility and actual delivery are not assessed.
    """
    if not pilot._finalized:
        raise RuntimeError('pilot_must_be_finalized_before_evaluation')
    expected_answer, expected_sources = _expected(case_id)
    try:
        answer = json.loads(response) if isinstance(response, str) else None
    except (TypeError, ValueError):
        answer = None
    events = pilot.audit()
    reads = [event for event in events if event.get('event') == 'read']
    observed = {event['source'] for event in reads if event.get('source') in expected_sources
                and _same_json_value(event.get('value'), expected_sources[event['source']])}
    mismatch = any(event.get('source') in expected_sources
                   and not _same_json_value(event.get('value'), expected_sources[event['source']]) for event in reads)
    return {
        'answer_matches_key': _same_json_value(answer, expected_answer),
        'required_sources_covered': observed == set(expected_sources) and not mismatch,
        'observed_required_sources': sorted(observed),
        'optional_status_read': any(event.get('source') == 'treatment-status' for event in reads),
        'index_read': any(event.get('source') == 'index' for event in reads),
        'source_mismatch': mismatch,
        'incomplete_http_request': any(event.get('event') == 'incomplete_http_request' for event in events),
        'candidate_status': status,
        'question_content': 'unassessed',
        'request_attribution': 'bearer_capability_only',
        'response_order': 'unassessed',
        'delivery': 'unassessed',
        'decision_qualified': False,
    }


def run(case_id: str, run_candidate: Callable[[str], Any]) -> dict[str, Any]:
    """Run one candidate against a fresh synthetic case, then score."""
    pilot = LoopbackPrescriptionPilot(sources(case_id))
    with pilot:
        outcome = run_candidate(pilot.url)
    return evaluate(pilot, case_id=case_id, response=outcome.response, status=outcome.status)
