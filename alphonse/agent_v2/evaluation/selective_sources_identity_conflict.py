"""Public development contrast requiring both identity and conflict resolution.

The records and source catalog are identical across both cases; only the
conversation's current identifier changes. The evaluator's expectations are
consulted only after the source service stops and never given to the candidate.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .loopback_prescription_pilot import LoopbackPrescriptionPilot, PrescriptionSources, _same_json_value
from .selective_sources_catalog_v2 import ALLOWED_SOURCES, GOAL, SOURCE_DESCRIPTION

CASE_IDS = ('current_verified', 'current_inconclusive')


def sources(case_id: str) -> PrescriptionSources:
    if case_id not in CASE_IDS:
        raise ValueError('unknown_identity_conflict_case')
    return PrescriptionSources(
        conversation={'current_prescription_id': 'rx-41' if case_id == 'current_verified' else 'rx-72'},
        prescription={'prescriptions': [
            {'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy'},
            {'id': 'rx-41', 'pharmacy': 'Maple Pharmacy'},
            {'id': 'rx-72', 'pharmacy': 'Oak Pharmacy'},
            {'id': 'rx-72', 'pharmacy': 'Willow Pharmacy'},
        ], 'verification_record': 'treatment-status'},
        treatment_status={'verifications': [
            {'prescription_id': 'rx-41', 'verification': 'verified', 'verified_current_pharmacy': 'Cedar Pharmacy'},
            {'prescription_id': 'rx-72', 'verification': 'inconclusive', 'verified_current_pharmacy': None},
        ]},
    )


def _expected(case_id: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Independently declare the development expectations, never model inputs."""
    if case_id not in CASE_IDS:
        raise ValueError('unknown_identity_conflict_case')
    known = case_id == 'current_verified'
    return ({'status': 'known' if known else 'unknown', 'pharmacy': 'Cedar Pharmacy' if known else None}, {
        'conversation': {'current_prescription_id': 'rx-41' if known else 'rx-72'},
        'prescription': {'prescriptions': [
            {'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy'}, {'id': 'rx-41', 'pharmacy': 'Maple Pharmacy'},
            {'id': 'rx-72', 'pharmacy': 'Oak Pharmacy'}, {'id': 'rx-72', 'pharmacy': 'Willow Pharmacy'},
        ], 'verification_record': 'treatment-status'},
        'treatment-status': {'verifications': [
            {'prescription_id': 'rx-41', 'verification': 'verified', 'verified_current_pharmacy': 'Cedar Pharmacy'},
            {'prescription_id': 'rx-72', 'verification': 'inconclusive', 'verified_current_pharmacy': None},
        ]},
    })


def evaluate(pilot: LoopbackPrescriptionPilot, *, case_id: str, response: Any, status: Any) -> dict[str, Any]:
    """Grade answer and matching observations separately after shutdown.

    Bearer audit does not establish reader identity or answer-formation order.
    Questions, actual delivery and clinical correctness are not assessed.
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
    """Run once on a fresh invented source service and score after shutdown."""
    pilot = LoopbackPrescriptionPilot(sources(case_id))
    with pilot:
        outcome = run_candidate(pilot.url)
    return evaluate(pilot, case_id=case_id, response=outcome.response, status=outcome.status)
