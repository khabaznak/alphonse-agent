"""Fixed clock development fixture and offline real-processor integration."""
from dataclasses import replace
import json
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from alphonse.agent_v2.core.inference import InferencePurpose, InferenceResult
from alphonse.agent_v2.evaluation.deterministic_adapters import _Action, _DeterministicProvider, _ReplayJev
from alphonse.agent_v2.evaluation.loopback_clock_pilot import LoopbackClockPilot, evaluate
from alphonse.agent_v2.evaluation.prescription_http_processors import run_candidate_http, run_v3_http

INSTANT = '2026-04-15T05:30:00+00:00'
ZONE = 'America/Mexico_City'
ANSWER = {'local_time': '23:30', 'local_date': '2026-04-14', 'timezone': ZONE}
GOAL = ('¿Qué hora es? Consulta la fuente de hora de esta prueba y responde con JSON: '
        'local_time (HH:MM), local_date (AAAA-MM-DD) y timezone.')


def read(url):
    with urlopen(url, timeout=3) as response:
        return json.load(response)


def test_separate_oracle_requires_read_and_handles_local_date():
    with LoopbackClockPilot(INSTANT, ZONE) as pilot:
        assert read(pilot.url) == {'instant_utc': INSTANT, 'timezone': ZONE}
        with pytest.raises(RuntimeError, match='finalized'):
            evaluate(pilot, response=ANSWER, expected_instant_utc=INSTANT, expected_timezone=ZONE)
        with pytest.raises(HTTPError):
            read(pilot.url + 'audit')
    result = evaluate(pilot, response=ANSWER, expected_instant_utc=INSTANT, expected_timezone=ZONE)
    assert result['passed'] and result['delivery'] == 'unassessed'
    assert result['response_order'] == 'unassessed' and result['decision_qualified'] is False
    assert pilot._capability not in repr(pilot.audit())
    assert not evaluate(pilot, response={**ANSWER, 'local_time': '05:30'}, expected_instant_utc=INSTANT, expected_timezone=ZONE)['passed']
    assert not evaluate(pilot, response=ANSWER, expected_instant_utc='2026-04-15T05:31:00+00:00', expected_timezone=ZONE)['passed']
    with pytest.raises(RuntimeError, match='reused'):
        with pilot:
            pass


def test_guessed_answer_cannot_pass_without_source_and_invalid_time_fails():
    with LoopbackClockPilot(INSTANT, ZONE) as pilot:
        pass
    assert not evaluate(pilot, response=ANSWER, expected_instant_utc=INSTANT, expected_timezone=ZONE)['passed']
    with pytest.raises(ValueError):
        LoopbackClockPilot('2026-04-15T05:30:00', ZONE)
    with pytest.raises(ValueError):
        LoopbackClockPilot(INSTANT, 'Not/A_Zone')


class Chooser:
    offline = True
    def __init__(self):
        self.actions = iter([
            {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'index'}},
            {'kind': 'answer', 'text': json.dumps(ANSWER)},
        ])
    def __call__(self, text, cap):
        decision = next(self.actions)
        return {**decision, 'usage': {'input_tokens': len(text.encode()),
                                      'output_tokens': len(json.dumps(decision).encode())}}


class Judge(_ReplayJev):
    offline = True


def provider_factory(state):
    state.scenario = replace(state.scenario,
        actions=(_Action('native.fixture_read', 'project_record_search', {'source': 'index'}, role='read_clock', behavior='read'),),
        final_response=json.dumps(ANSWER))
    class Provider(_DeterministicProvider):
        def generate_json(self, request):
            if request.purpose == InferencePurpose.TASK_UNDERSTANDING:
                return InferenceResult(json_value={'requested_outcome': GOAL, 'supplied_parameters': [],
                    'constraints': [], 'unresolved_dependencies': [], 'superseded_constraints': []})
            return super().generate_json(request)
    return Provider(state)
provider_factory.offline = True


@pytest.mark.parametrize('arm', ['candidate', 'current_v3', 'targeted_v3'])
def test_actual_processor_reads_common_http_clock(monkeypatch, arm):
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', '1' if arm == 'targeted_v3' else '0')
    events = []
    with LoopbackClockPilot(INSTANT, ZONE) as pilot:
        if arm == 'candidate':
            result = run_candidate_http(base_url=pilot.url, goal=GOAL, chooser=Chooser(), emit=events.append)
        else:
            result = run_v3_http(base_url=pilot.url, goal=GOAL, provider_factory=provider_factory,
                                 jev=Judge(), emit=events.append)
    assert result.status == 'completed', result.reason
    verdict = evaluate(pilot, response=json.loads(result.response), expected_instant_utc=INSTANT, expected_timezone=ZONE)
    assert verdict['passed'] is True and verdict['decision_qualified'] is False
    assert [event['source'] for event in events if event['kind'] == 'source_read'] == ['index']
