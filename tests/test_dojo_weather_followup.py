"""Exercise fixture grading and sequential follow-up orchestration."""
import hashlib
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.evaluation.dojo_weather_followup import REPORT, scenario


@pytest.mark.parametrize('language', ['en', 'es'])
def test_weather_scenario_submits_followups_in_order_and_keeps_scoring_honest(language):
    case = scenario(language)
    prompts = []
    answers = ['Campeche: 31 °C', 'Oaxaca: 18 °C']

    class Client:
        def request(self, method, **kwargs):
            if method == 'experimental_owner_ingress':
                prompts.append(kwargs['prompt'])
                return {'message_id': str(len(prompts))}
            if method == 'experimental_owner_status':
                return {'status': 'ready'}
            if method == 'experimental_owner_take_draft':
                return {'message': answers[int(kwargs['message_id']) - 1],
                        'kind': 'answer', 'acknowledgement_token': 'token'}
            if method == 'experimental_owner_ack_draft':
                return {'status': 'client_acknowledged'}
            raise AssertionError(method)

    runtime = SimpleNamespace(conversation_store=SimpleNamespace(list=lambda **_: [
        SimpleNamespace(role='user', content=case.GOAL),
        SimpleNamespace(role='assistant', content='Mérida: 23 °C'),
    ]))
    events = []
    result = case.after_owner(None, runtime, 'owner', events.append, client=Client())
    assert result['followup_outcome'] == 'observed'
    assert len(prompts) == len(events) == 2
    assert len(result['turns']) == 3
    grade = case.grade({'status': 'observed', 'sha256': hashlib.sha256(REPORT.encode()).hexdigest(),
                        'after_owner': result})
    assert grade['fixture_integrity'] == 'unchanged'
    assert all(row['expected_temperature_mentioned'] for row in grade['turn_checks'])
    assert all(row['expected_city_mentioned'] for row in grade['turn_checks'])
    assert grade['effect_passed'] is None
    assert grade['response_semantics'] == 'requires_manual_review'


def test_weather_scenario_does_not_invent_followup_when_first_turn_missing():
    case = scenario('en')
    runtime = SimpleNamespace(conversation_store=SimpleNamespace(list=lambda **_: [
        SimpleNamespace(role='user', content=case.GOAL)]))
    result = case.after_owner(None, runtime, 'owner', lambda _: None, client=None)
    assert result == {'turns': [], 'followup_outcome': 'first_turn_unavailable'}
    grade = case.grade({'status': 'unavailable', 'after_owner': result})
    assert grade['effect_passed'] is None
    assert grade['fixture_integrity'] == 'unavailable'
    assert all(row['status'] == 'not_attempted' for row in grade['turn_checks'])
