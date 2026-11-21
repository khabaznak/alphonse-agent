"""Ensure the appointment exercise keeps incomplete and semantic results honest."""
from types import SimpleNamespace

from alphonse.agent_v2.evaluation.dojo_plan_followup import FOLLOWUPS, GOAL, scenario


def test_plan_followup_submits_both_turns_and_keeps_semantics_unassessed():
    case = scenario()
    prompts = []
    class Client:
        def request(self, method, **params):
            if method == 'experimental_owner_ingress':
                prompts.append(params['prompt'])
                return {'message_id': str(len(prompts))}
            if method == 'experimental_owner_status':
                return {'status': 'ready'}
            if method == 'experimental_owner_take_draft':
                text = ['Thursday at 10 a.m.', 'Tuesday at 3 p.m.'][int(params['message_id']) - 1]
                return {'message': text, 'kind': 'answer', 'acknowledgement_token': 'token'}
            if method == 'experimental_owner_ack_draft':
                return {'status': 'client_acknowledged'}
            raise AssertionError(method)
    runtime = SimpleNamespace(conversation_store=SimpleNamespace(list=lambda **_: [
        SimpleNamespace(role='user', content=GOAL),
        SimpleNamespace(role='assistant', content='Tuesday at 3 p.m.')]))
    after = case.after_owner(None, runtime, 'owner', lambda _: None, client=Client())
    assert prompts == list(FOLLOWUPS)
    assert len(after['turns']) == 3 and after['followup_outcome'] == 'observed'
    grade = case.grade({'after_owner': after})
    assert all(t['expected_weekday_mentioned'] and t['expected_hour_mentioned']
               for t in grade['turn_checks'])
    assert grade['effect_passed'] is None
    assert grade['response_semantics'] == 'requires_manual_review'


def test_plan_followup_stops_after_first_turn_fails():
    case = scenario()
    runtime = SimpleNamespace(conversation_store=SimpleNamespace(list=lambda **_: [
        SimpleNamespace(role='user', content=GOAL)]))
    after = case.after_owner(None, runtime, 'owner', lambda _: None, client=None)
    assert after == {'turns': [], 'followup_outcome': 'first_turn_unavailable'}
    assert case.grade({'after_owner': after})['turn_checks'][0]['status'] == 'not_attempted'
