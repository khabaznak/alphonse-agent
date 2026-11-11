"""Real V3 + Responses parser + fixed clock, with fake providers only."""
import json

import pytest

from alphonse.agent_v2.evaluation.loopback_clock_pilot import LoopbackClockPilot, evaluate
from alphonse.agent_v2.evaluation.v3_clock_responses import run_v3_clock_offline

INSTANT = '2026-04-15T05:30:00+00:00'
ZONE = 'America/Mexico_City'
GOAL = '¿Qué hora es? Consulta la fuente sintética.'
ANSWER = {'local_time': '23:30', 'local_date': '2026-04-14', 'timezone': ZONE}


@pytest.mark.parametrize('flag', [False, True])
def test_actual_v3_responses_worker_reads_independent_clock(flag, monkeypatch):
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', '1' if not flag else '0')
    with LoopbackClockPilot(INSTANT, ZONE) as pilot:
        result = run_v3_clock_offline(base_url=pilot.url, goal=GOAL, scripted_answer=ANSWER,
                                      projection_enabled=flag)
    assert result['availability'] == 'available', result
    assert result['projection_enabled'] is flag
    assert result['processor']['status'] == 'completed', result
    assert result['decision_qualified'] is False and result['delivery'] == 'unassessed'
    assert result['ledger']['calls'] > 0
    assert any(e.get('kind') == 'source_read' for e in result['observations'])
    assert evaluate(pilot, response=json.loads(result['processor']['response']),
                    expected_instant_utc=INSTANT, expected_timezone=ZONE)['passed']
    assert pilot._capability not in repr(result)


def test_wrong_scripted_answer_fails_independent_oracle():
    with LoopbackClockPilot(INSTANT, ZONE) as pilot:
        result = run_v3_clock_offline(base_url=pilot.url, goal=GOAL,
            scripted_answer={**ANSWER, 'local_time': '05:30'}, projection_enabled=False)
    assert result['availability'] == 'available' and result['processor']['status'] == 'completed'
    assert not evaluate(pilot, response=json.loads(result['processor']['response']),
                        expected_instant_utc=INSTANT, expected_timezone=ZONE)['passed']


def test_invalid_configuration_fails_before_worker():
    with pytest.raises(ValueError, match='configuration'):
        run_v3_clock_offline(base_url='x', goal=GOAL, scripted_answer=ANSWER, projection_enabled='yes')
