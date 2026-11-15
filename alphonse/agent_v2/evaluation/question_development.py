"""Matched scripted question replay; it does not deliver real notifications."""
from __future__ import annotations

import json
from .decision_followthrough_development import Case
from .followthrough_candidate_adapter import run_candidate_fixture


def compare_missing_identity(*, simulate_delivery=False):
    """Exercise the simple loop's lookup and necessary question.

    The candidate chooses its question decision. A simulated bridge is optional and is not product
    delivery. Provider decisions are preselected, not autonomous reasoning.
    """
    if type(simulate_delivery) is not bool:
        raise ValueError('simulate_delivery_must_be_boolean')
    case = Case('missing')
    text = json.dumps({'kind': 'handoff', 'blocker': 'missing_identity', 'known': [],
                       'unknown': ['appointment_time'], 'question': 'appointment_identity'})
    def candidate():
        script = iter([{'kind': 'tool', 'tool_id': 'native.fixture_lookup',
                        'arguments': {'source': 'primary'}}, {'kind': 'question', 'text': text}])
        return run_candidate_fixture(case, lambda _text: next(script), simulate_delivery=simulate_delivery)
    arms = {
        'single_loop': candidate(),
    }
    return {'arms': arms, 'decision_qualified': False, 'delivery': 'synthetic_bridge' if simulate_delivery else 'unassessed',
            'limitations': ['Scripted decisions do not measure autonomous question choice.',
                            'Bridge is outside processors and does not prove product delivery.',
                            'Permission denial with no question is not evaluated by this scenario.']}
