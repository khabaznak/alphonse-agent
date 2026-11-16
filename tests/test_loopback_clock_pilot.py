"""Fixed clock development fixture and offline real-processor integration."""
import json
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from alphonse.agent_v2.evaluation.loopback_clock_pilot import LoopbackClockPilot, evaluate
from alphonse.agent_v2.evaluation.prescription_http_processors import run_candidate_http

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


def test_same_instant_in_different_zones_requires_both_time_and_date_to_change():
    """A clock answer must follow the observed zone, not a memorized fixture."""
    tokyo_zone = 'Asia/Tokyo'
    tokyo_answer = {'local_time': '14:30', 'local_date': '2026-04-15', 'timezone': tokyo_zone}
    with LoopbackClockPilot(INSTANT, ZONE) as mexico:
        assert read(mexico.url)['timezone'] == ZONE
    with LoopbackClockPilot(INSTANT, tokyo_zone) as tokyo:
        assert read(tokyo.url)['timezone'] == tokyo_zone

    assert evaluate(mexico, response=ANSWER, expected_instant_utc=INSTANT, expected_timezone=ZONE)['passed']
    assert evaluate(tokyo, response=tokyo_answer, expected_instant_utc=INSTANT,
                    expected_timezone=tokyo_zone)['passed']
    # Both observations are valid reads; swapping the answers must still fail.
    wrong_mexico = evaluate(mexico, response=tokyo_answer, expected_instant_utc=INSTANT, expected_timezone=ZONE)
    wrong_tokyo = evaluate(tokyo, response=ANSWER, expected_instant_utc=INSTANT, expected_timezone=tokyo_zone)
    assert wrong_mexico['clock_source_matches'] and not wrong_mexico['answer_matches_key']
    assert wrong_tokyo['clock_source_matches'] and not wrong_tokyo['answer_matches_key']
    assert not wrong_mexico['passed'] and not wrong_tokyo['passed']


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


def test_simple_loop_reads_http_clock():
    events = []
    with LoopbackClockPilot(INSTANT, ZONE) as pilot:
        result = run_candidate_http(base_url=pilot.url, goal=GOAL, chooser=Chooser(), emit=events.append)
    assert result.status == 'completed', result.reason
    verdict = evaluate(pilot, response=json.loads(result.response), expected_instant_utc=INSTANT, expected_timezone=ZONE)
    assert verdict['passed'] is True and verdict['decision_qualified'] is False
    assert [event['source'] for event in events if event['kind'] == 'source_read'] == ['index']
