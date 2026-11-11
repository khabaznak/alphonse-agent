"""Matched synthetic denial replay; not product permission or delivery proof."""
from __future__ import annotations

import json

from .decision_followthrough_development import Case
from .followthrough_candidate_adapter import run_candidate_fixture
from .followthrough_v3_adapter import run_v3_fixture


def compare_denied(*, simulate_delivery: bool = False) -> dict:
    """Replay an authorized source reporting denial, then a truthful handoff.

    The expected response and decisions are supplied to the scripted chooser.
    This does not test the processor's production actor-scoped authorization;
    the fixture source itself reports the denial. The optional bridge runs
    after processing and is not the product's delivery mechanism.
    """
    if type(simulate_delivery) is not bool:
        raise ValueError('simulate_delivery_must_be_boolean')
    case = Case('denied')
    response = json.dumps({'kind': 'handoff', 'blocker': 'denied', 'known': [],
                           'unknown': ['appointment_time'], 'question': None})

    def candidate():
        steps = iter([{'kind': 'tool', 'tool_id': 'native.fixture_lookup',
                       'arguments': {'source': 'primary'}},
                      {'kind': 'answer', 'text': response}])
        result = run_candidate_fixture(case, lambda _: next(steps), simulate_delivery=simulate_delivery)
        # Completion here means the synthetic response was prepared, not that
        # the appointment was found or the user was actually notified.
        outcome = result['fixture_outcome']
        outcome['terminal_semantics'] = result['processor_status'] == 'completed'
        outcome['passed'] = outcome['passed'] and outcome['terminal_semantics']
        return result

    arms = {
        'current_v3': run_v3_fixture(case, projection_enabled=False, scripted_sources=['primary'],
                                     scripted_response=response, simulate_delivery=simulate_delivery),
        'targeted_v3_projection': run_v3_fixture(case, projection_enabled=True, scripted_sources=['primary'],
                                     scripted_response=response, simulate_delivery=simulate_delivery),
        'single_loop': candidate(),
    }
    return {'arms': arms, 'decision_qualified': False,
            'delivery': 'synthetic_bridge' if simulate_delivery else 'unassessed',
            'limitations': ['Preselected actions and text do not demonstrate autonomous recovery.',
                            'A fixture-reported denial is not a production authorization boundary.',
                            'A prepared response is not actual user delivery.',
                            'A truthful handoff is not a verified task completion.']}
