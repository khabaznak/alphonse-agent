"""Synthetic multi-source ambiguity case for the development gym.

A current prescription identifier is available in one record; another lists
conflicting pharmacy values for that identifier. Expected records and answer
are applied only after shutdown, never exposed as a hidden runtime requirement.
This is neither a clinical exercise nor a held-out benchmark.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .loopback_prescription_pilot import LoopbackPrescriptionPilot, PrescriptionSources, _same_json_value
from .selective_sources_development import ALLOWED_SOURCES, SOURCE_DESCRIPTION

GOAL = ('Which pharmacy is associated with the current prescription in the synthetic records? '
        'Return a JSON object with status "known" or "unknown", and pharmacy as a string '
        'when known or null when unknown.')


def sources() -> PrescriptionSources:
    """Create invented records, with no real household data."""
    return PrescriptionSources(
        conversation={'current_prescription_id': 'rx-41'},
        prescription={'prescriptions': [
            {'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy'},
            {'id': 'rx-41', 'pharmacy': 'Maple Pharmacy'},
        ]},
        treatment_status={'exercise_program': 'stretching', 'status': 'paused'},
    )


def evaluate(pilot: LoopbackPrescriptionPilot, *, response: Any, status: Any) -> dict[str, Any]:
    """Grade exact uncertainty and verified source reads separately after shutdown.

    A waiting-user outcome is reported independently: a question can be useful,
    but this fixture cannot semantically grade its content or establish delivery.
    Audit reads do not establish process identity or answer-formation order.
    """
    if not pilot._finalized:
        raise RuntimeError('pilot_must_be_finalized_before_evaluation')
    try:
        answer = json.loads(response) if isinstance(response, str) else None
    except (TypeError, ValueError):
        answer = None
    expected_answer = {'status': 'unknown', 'pharmacy': None}
    expected = {
        'conversation': {'current_prescription_id': 'rx-41'},
        'prescription': {'prescriptions': [
            {'id': 'rx-41', 'pharmacy': 'Cedar Pharmacy'},
            {'id': 'rx-41', 'pharmacy': 'Maple Pharmacy'},
        ]},
    }
    events = pilot.audit()
    reads = [e for e in events if e.get('event') == 'read']
    observed = {e['source'] for e in reads if e.get('source') in expected
                and _same_json_value(e.get('value'), expected[e['source']])}
    mismatch = any(e.get('source') in expected and not _same_json_value(e.get('value'), expected[e['source']])
                   for e in reads)
    return {
        'answer_matches_key': _same_json_value(answer, expected_answer),
        'required_sources_covered': observed == set(expected) and not mismatch,
        'observed_required_sources': sorted(observed),
        'unrelated_source_read': any(e.get('source') == 'treatment-status' for e in reads),
        'index_read': any(e.get('source') == 'index' for e in reads),
        'source_mismatch': mismatch,
        'incomplete_http_request': any(e.get('event') == 'incomplete_http_request' for e in events),
        'candidate_status': status,
        'question_content': 'unassessed',
        'request_attribution': 'bearer_capability_only',
        'response_order': 'unassessed',
        'delivery': 'unassessed',
        'decision_qualified': False,
    }


def run(run_candidate: Callable[[str], Any]) -> dict[str, Any]:
    """Run once against the invented records and grade after service shutdown."""
    pilot = LoopbackPrescriptionPilot(sources())
    with pilot:
        outcome = run_candidate(pilot.url)
    return evaluate(pilot, response=outcome.response, status=outcome.status)
