"""Development-only follow-up for identifying the requested record.

The fixture is identical to the public source-selection variants. The answer
diagnostic is computed after service shutdown and retains only categorical
shape information, never the candidate's response or source contents.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .loopback_prescription_pilot import LoopbackPrescriptionPilot
from .selective_sources_catalog_v2 import ALLOWED_SOURCES, CASE_IDS, GOAL, SOURCE_DESCRIPTION
from .selective_sources_variants import evaluate, sources


def answer_shape(response: Any) -> dict[str, str]:
    """Describe only fixed categories without retaining generated values."""
    if not isinstance(response, str) or not response.strip():
        return {'format': 'no_answer'}
    try:
        answer = json.loads(response)
    except (TypeError, ValueError):
        return {'format': 'invalid_json'}
    if not isinstance(answer, dict):
        return {'format': 'not_object'}
    status = answer.get('status')
    pharmacy = answer.get('pharmacy')
    return {
        'format': 'object',
        'status': ('missing' if 'status' not in answer else
                   'invalid_type' if not isinstance(status, str) else
                   status if status in ('known', 'unknown') else 'other'),
        'pharmacy': ('missing' if 'pharmacy' not in answer else
                     'null' if pharmacy is None else
                     'string' if isinstance(pharmacy, str) else 'other_type'),
    }


def run(case_id: str, run_candidate: Callable[[str], Any], *,
        question_assessor: Callable[[str, str, tuple[str, ...]], str] | None = None) -> dict[str, Any]:
    """Run a candidate and optionally assess a synthetic question after shutdown.

    An assessor receives the public development case ID, the transient
    question text, and observed source names; it must be authorized for that synthetic content and should
    be independent of the candidate. Its categorical judgment is not verified
    by this module. The assessor is never passed to the candidate or invoked
    while the source service is running. Question text is not returned or
    deliberately stored here, but an assessor may retain it. This measures a
    proposed question, not actual delivery or successful user follow-through.
    """
    if question_assessor is not None and not callable(question_assessor):
        raise ValueError('invalid_question_assessor')
    pilot = LoopbackPrescriptionPilot(sources(case_id))
    with pilot:
        outcome = run_candidate(pilot.url)
    result = evaluate(pilot, case_id=case_id, response=outcome.response, status=outcome.status)
    result['answer_shape'] = answer_shape(outcome.response)
    observed = tuple(sorted({e.get('source') for e in pilot.audit()
                             if e.get('event') == 'read' and e.get('source') in ALLOWED_SOURCES}))
    result['question_assessment'] = assess_question(case_id, outcome.status, outcome.response,
                                                    question_assessor, observed_sources=observed)
    if result['question_assessment']['basis'] == 'caller_assessor_unverified':
        result['question_content'] = 'assessed_by_caller'
    return result


def assess_question(case_id: str, status: Any, response: Any,
                    assessor: Callable[[str, str, tuple[str, ...]], str] | None, *,
                    observed_sources: tuple[str, ...] = ()) -> dict[str, str]:
    """Return only a bounded, caller-supplied judgment, never question text."""
    if status != 'waiting_user':
        return {'status': 'not_asked', 'basis': 'candidate_status'}
    if not isinstance(response, str) or not response.strip():
        return {'status': 'unassessed', 'basis': 'missing_question'}
    if assessor is None:
        return {'status': 'unassessed', 'basis': 'no_assessor'}
    try:
        judgment = assessor(case_id, response, observed_sources)
    except Exception:
        return {'status': 'unassessed', 'basis': 'assessor_failed'}
    if type(judgment) is not str or judgment not in {'useful', 'not_useful', 'unclear'}:
        return {'status': 'unassessed', 'basis': 'invalid_assessment'}
    return {'status': judgment, 'basis': 'caller_assessor_unverified'}
