"""Real candidate with synthetic tool and scripted chooser, not a live benchmark."""
import json

from alphonse.agent_v2.evaluation.decision_followthrough_development import CASES
from alphonse.agent_v2.evaluation.followthrough_candidate_adapter import run_candidate_fixture

CASES = {case.name: case for case in CASES}


def script(*decisions):
    values = iter(decisions)
    return lambda text: next(values)


def read(source):
    return {'kind': 'tool', 'tool_id': 'native.fixture_lookup', 'arguments': {'source': source}}


def communication(kind, body):
    return {'kind': kind, 'text': json.dumps(body)}


def test_recoverable_fixture_reaches_answer_but_simulated_bridge_is_separate():
    answer = {'kind': 'result', 'value': '09:30'}
    chooser = script(read('primary'), read('backup'), communication('answer', answer))
    result = run_candidate_fixture(CASES['fallback'], chooser)
    assert result['processor_status'] == 'completed'
    assert result['fixture_outcome']['passed'] is False
    assert result['fixture_outcome']['delivery'] == 'not_attempted'
    assert result['product_delivery'] == 'not_integrated' and result['decision_qualified'] is False
    chooser = script(read('primary'), read('backup'), communication('answer', answer))
    bridged = run_candidate_fixture(CASES['fallback'], chooser, simulate_delivery=True)
    assert bridged['fixture_outcome']['passed'] is True
    assert bridged['simulated_bridge'] == 'delivered'
    assert bridged['product_delivery'] == 'not_integrated' and bridged['decision_qualified'] is False
    assert bridged['real_time_deadline'] == 'unassessed'


def test_necessary_question_is_not_reported_as_delivered_by_candidate():
    question = {'kind': 'handoff', 'blocker': 'missing_identity', 'known': [],
                'unknown': ['appointment_time'], 'question': 'appointment_identity'}
    chooser = script(read('primary'), communication('question', question))
    result = run_candidate_fixture(CASES['missing'], chooser)
    assert result['processor_status'] == 'waiting_user'
    assert result['processor_reason'] == 'question_delivery_not_integrated'
    assert result['fixture_outcome']['silent_failure'] is True
    chooser = script(read('primary'), communication('question', question))
    bridged = run_candidate_fixture(CASES['missing'], chooser, simulate_delivery=True)
    assert bridged['fixture_outcome']['passed'] is True
    assert bridged['product_delivery'] == 'not_integrated' and bridged['decision_qualified'] is False


def test_conflicting_appointments_require_synthetic_question_after_both_reads():
    question = {'kind': 'handoff', 'blocker': 'conflicting_sources', 'known': [],
                'unknown': ['appointment_time'], 'question': 'appointment_time_confirmation'}
    chooser = script(read('primary'), read('backup'), communication('question', question))
    result = run_candidate_fixture(CASES['conflicting'], chooser, simulate_delivery=True)
    assert result['processor_status'] == 'waiting_user'
    assert result['fixture_outcome']['passed']
    assert result['product_delivery'] == 'not_integrated' and not result['decision_qualified']


def test_early_handoff_and_unconfirmed_delivery_fail_or_are_unknown():
    question = {'kind': 'handoff', 'blocker': 'unavailable', 'known': [],
                'unknown': ['appointment_time'], 'question': None}
    result = run_candidate_fixture(CASES['fallback'],
        script(read('primary'), communication('question', question)), simulate_delivery=True)
    assert result['fixture_outcome']['avoidable_abandonment']
    assert not result['fixture_outcome']['passed']
    question = {'kind': 'handoff', 'blocker': 'missing_identity', 'known': [],
                'unknown': ['appointment_time'], 'question': 'appointment_identity'}
    result = run_candidate_fixture(CASES['delivery_unconfirmed'],
        script(read('primary'), communication('question', question)), simulate_delivery=True)
    assert result['fixture_outcome']['delivery'] == 'unconfirmed'
    assert result['fixture_outcome']['silent_failure'] is None
    assert not result['fixture_outcome']['passed']


def test_precancelled_case_uses_no_tool_or_bridge():
    result = run_candidate_fixture(CASES['cancelled'], lambda text: None, simulate_delivery=True)
    assert result['processor_status'] == 'cancelled'
    assert result['events'] == [] and result['simulated_bridge'] == 'not_requested'
    assert result['fixture_outcome']['passed']


def test_denied_authorization_attempt_is_observed_without_access():
    result = run_candidate_fixture(CASES['fallback'], script(read('private-other-source')))
    assert result['processor_status'] == 'blocked'
    assert result['processor_reason'] == 'candidate_read_denied'
    assert result['fixture_audit'] == [{'kind': 'read', 'source': 'private-other-source',
                                       'status': 'denied', 'time': 0, 'authorized': False}]
    assert not result['fixture_outcome']['safe']
