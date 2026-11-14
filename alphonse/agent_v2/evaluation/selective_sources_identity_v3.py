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


def run(case_id: str, run_candidate: Callable[[str], Any]) -> dict[str, Any]:
    """Run a single candidate and derive its assessment after shutdown."""
    pilot = LoopbackPrescriptionPilot(sources(case_id))
    with pilot:
        outcome = run_candidate(pilot.url)
    result = evaluate(pilot, case_id=case_id, response=outcome.response, status=outcome.status)
    result['answer_shape'] = answer_shape(outcome.response)
    return result
