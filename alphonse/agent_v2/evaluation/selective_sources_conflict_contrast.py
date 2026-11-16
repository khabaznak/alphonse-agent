"""Public synthetic contrast: a third record may or may not resolve conflict.

The candidate receives only the task, source descriptions and tool results.
Independent expectations are used after shutdown, never as source hints. This
is an exploratory development exercise, not medical or held-out evaluation.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .loopback_prescription_pilot import LoopbackPrescriptionPilot, PrescriptionSources, _same_json_value
from .selective_sources_catalog_v2 import ALLOWED_SOURCES, GOAL, SOURCE_DESCRIPTION

CASE_IDS = ('resolved', 'unresolved')


def sources(case_id: str) -> PrescriptionSources:
    if case_id not in CASE_IDS:
        raise ValueError('unknown_conflict_contrast_case')
    status = ({'prescription_id': 'rx-41', 'verification': 'verified',
               'verified_current_pharmacy': 'Cedar Pharmacy'} if case_id == 'resolved' else
              {'prescription_id': 'rx-41', 'verification': 'inconclusive',
               'verified_current_pharmacy': None})
    return PrescriptionSources(
        conversation={'current_prescription_id': 'rx-41'},
        prescription={'prescriptions': [
            {'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy'},
            {'id': 'rx-41', 'pharmacy': 'Maple Pharmacy'},
            {'id': 'rx-72', 'pharmacy': 'Willow Pharmacy'},
        ], 'verification_record': 'treatment-status'},
        treatment_status=status,
    )


def _expected(case_id: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Declare expected response and observed values independently of sources()."""
    if case_id not in CASE_IDS:
        raise ValueError('unknown_conflict_contrast_case')
    response = ({'status': 'known', 'pharmacy': 'Cedar Pharmacy'} if case_id == 'resolved'
                else {'status': 'unknown', 'pharmacy': None})
    status = ({'prescription_id': 'rx-41', 'verification': 'verified',
               'verified_current_pharmacy': 'Cedar Pharmacy'} if case_id == 'resolved' else
              {'prescription_id': 'rx-41', 'verification': 'inconclusive',
               'verified_current_pharmacy': None})
    return response, {
        'conversation': {'current_prescription_id': 'rx-41'},
        'prescription': {'prescriptions': [
            {'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy'},
            {'id': 'rx-41', 'pharmacy': 'Maple Pharmacy'},
            {'id': 'rx-72', 'pharmacy': 'Willow Pharmacy'},
        ], 'verification_record': 'treatment-status'},
        'treatment-status': status,
    }


def evaluate(pilot: LoopbackPrescriptionPilot, *, case_id: str, response: Any, status: Any) -> dict[str, Any]:
    """Score response and source coverage independently after service shutdown.

    The bearer audit cannot establish reader identity or when the answer was
    formed. A waiting-user question and actual user delivery are unassessed.
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
                   and not _same_json_value(event.get('value'), expected_sources[event['source']])
                   for event in reads)
    return {
        'answer_matches_key': _same_json_value(answer, expected_answer),
        'required_sources_covered': observed == set(expected_sources) and not mismatch,
        'observed_required_sources': sorted(observed),
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
    """Run a candidate against a fresh disposable service, then evaluate."""
    pilot = LoopbackPrescriptionPilot(sources(case_id))
    with pilot:
        outcome = run_candidate(pilot.url)
    return evaluate(pilot, case_id=case_id, response=outcome.response, status=outcome.status)
