"""Synthetic contrast: the former distractor is relevant to a different goal.

This development case reuses the same source service and descriptions as the
pharmacy exercise. Its scoring key is consulted only after service shutdown;
neither the key nor a preferred-source list is passed to the candidate.
"""
from __future__ import annotations

from typing import Any, Callable

from .loopback_prescription_pilot import LoopbackPrescriptionPilot, _same_json_value
from .selective_sources_development import ALLOWED_SOURCES, SOURCE_DESCRIPTION, sources

GOAL = ('Which exercise program is currently paused in the synthetic records? '
        'Answer with only the program name.')


def evaluate(pilot: LoopbackPrescriptionPilot, *, response: Any, status: Any) -> dict[str, Any]:
    """Score the answer and relevant and irrelevant reads independently.

    The expected record is specified separately from the fixture source. The
    audit cannot establish process identity or the ordering of answer formation.
    """
    if not pilot._finalized:
        raise RuntimeError('pilot_must_be_finalized_before_evaluation')
    events = pilot.audit()
    reads = [event for event in events if event.get('event') == 'read']
    status_reads = [event for event in reads if event.get('source') == 'treatment-status']
    expected = {'exercise_program': 'stretching', 'status': 'paused'}
    mismatch = any(not _same_json_value(event.get('value'), expected) for event in status_reads)
    return {
        'answer_matches_key': isinstance(response, str) and response.strip().casefold() == 'stretching',
        'required_source_covered': bool(status_reads) and not mismatch,
        'source_mismatch': mismatch,
        'irrelevant_sources_read': sorted({event['source'] for event in reads
                                           if event.get('source') in {'prescription', 'conversation'}}),
        'index_read': any(event.get('source') == 'index' for event in reads),
        'incomplete_http_request': any(event.get('event') == 'incomplete_http_request' for event in events),
        'candidate_status': status,
        'request_attribution': 'bearer_capability_only',
        'response_order': 'unassessed',
        'delivery': 'unassessed',
        'decision_qualified': False,
    }


def run(run_candidate: Callable[[str], Any]) -> dict[str, Any]:
    """Run once on synthetic sources and evaluate after the service shuts down."""
    pilot = LoopbackPrescriptionPilot(sources())
    with pilot:
        outcome = run_candidate(pilot.url)
    return evaluate(pilot, response=outcome.response, status=outcome.status)
