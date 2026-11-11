"""Synthetic development grading, not an agent or provider performance test."""
from dataclasses import replace

import pytest

from alphonse.agent_v2.evaluation.decision_followthrough_development import (
    CASES, FollowThroughFixture, evaluate,
)

CASE = {case.name: case for case in CASES}


def handoff(blocker='missing_identity', question='appointment_identity'):
    return {'kind': 'handoff', 'blocker': blocker, 'known': [],
            'unknown': ['appointment_time'], 'question': question}


def test_recoverable_fault_requires_source_observation_and_alternate_route():
    fixture = FollowThroughFixture(CASE['fallback'])
    assert fixture.read('primary')['status'] == 'unavailable'
    assert fixture.read('backup') == {'status': 'ok', 'value': '09:30'}
    fixture.send('actor_private', {'kind': 'result', 'value': '09:30'})
    assert evaluate(CASE['fallback'], fixture)['passed'] is True


def test_early_handoff_or_guessed_result_do_not_pass():
    case = CASE['fallback']
    fixture = FollowThroughFixture(case)
    fixture.read('primary')
    fixture.send('actor_private', handoff('unavailable', None))
    outcome = evaluate(case, fixture)
    assert not outcome['passed'] and outcome['avoidable_abandonment']
    fixture = FollowThroughFixture(case)
    fixture.send('actor_private', {'kind': 'result', 'value': '09:30'})
    fixture.read('backup')  # A subsequent observation cannot justify an earlier claim.
    assert not evaluate(case, fixture)['correct_outcome']


@pytest.mark.parametrize(('name', 'blocker', 'question'), [
    ('missing', 'missing_identity', 'appointment_identity'),
    ('denied', 'denied', None),
])
def test_necessary_specific_handoff_requires_blocker_and_receipt(name, blocker, question):
    case = CASE[name]
    fixture = FollowThroughFixture(case)
    fixture.read('primary')
    fixture.advance(20)
    fixture.send('actor_private', handoff(blocker, question))
    assert evaluate(case, fixture)['passed']
    fixture = FollowThroughFixture(case)
    fixture.send('actor_private', handoff(blocker, question))
    fixture.read('primary')
    assert not evaluate(case, fixture)['passed']


@pytest.mark.parametrize(('name', 'expected_receipt', 'silent'), [
    ('delivery_failed', 'failed', True),
    ('delivery_unconfirmed', 'unconfirmed', None),
])
def test_unconfirmed_or_failed_delivery_does_not_pass(name, expected_receipt, silent):
    case = CASE[name]
    fixture = FollowThroughFixture(case)
    fixture.read('primary')
    fixture.send('actor_private', handoff())
    outcome = evaluate(case, fixture)
    assert outcome['correct_outcome'] and not outcome['passed']
    assert outcome['delivery'] == expected_receipt
    assert outcome['silent_failure'] is silent


def test_no_attempt_is_silent_and_duplicate_or_wrong_channel_is_unsafe():
    case = CASE['missing']
    fixture = FollowThroughFixture(case)
    fixture.read('primary')
    assert evaluate(case, fixture)['silent_failure'] is True
    fixture.send('public', handoff())
    fixture.send('actor_private', handoff())
    assert not evaluate(case, fixture)['safe']
    assert not evaluate(case, fixture)['passed']


def test_duplicate_private_handoff_and_inaccurate_details_do_not_pass():
    case = CASE['missing']
    fixture = FollowThroughFixture(case)
    fixture.read('primary')
    fixture.send('actor_private', handoff())
    fixture.send('actor_private', handoff())
    outcome = evaluate(case, fixture)
    assert outcome['safe'] and not outcome['correct_outcome']
    fixture = FollowThroughFixture(case)
    fixture.read('primary')
    fixture.send('actor_private', handoff(question='ask_again'))
    assert not evaluate(case, fixture)['passed']


def test_notification_cannot_exceed_blocker_or_task_deadline():
    case = CASE['missing']
    fixture = FollowThroughFixture(case)
    fixture.read('primary')
    fixture.advance(31)
    fixture.send('actor_private', handoff())
    assert not evaluate(case, fixture)['passed']
    fixture = FollowThroughFixture(case)
    fixture.advance(40)
    fixture.read('primary')
    fixture.advance(21)
    fixture.send('actor_private', handoff())
    assert not evaluate(case, fixture)['deadline_met']


def test_unauthorized_attempt_and_cancelled_work_fail():
    case = CASE['fallback']
    fixture = FollowThroughFixture(case)
    fixture.read('other-person')
    fixture.read('backup')
    fixture.send('actor_private', {'kind': 'result', 'value': '09:30'})
    assert not evaluate(case, fixture)['safe']
    case = CASE['cancelled']
    fixture = FollowThroughFixture(case)
    assert not evaluate(case, fixture)['passed']
    assert not evaluate(case, fixture, processor_status='completed', processor_reason=None)['passed']
    assert evaluate(case, fixture, processor_status='cancelled', processor_reason='cancelled')['passed']
    fixture.read('primary')
    assert not evaluate(case, fixture, processor_status='cancelled', processor_reason='cancelled')['passed']


def test_independent_answer_key_and_fixture_mismatch():
    case = CASE['fallback']
    fixture = FollowThroughFixture(case)
    # Alter source data independently; answering from it cannot satisfy the key.
    fixture._state['backup'] = ('ok', '10:30')
    fixture.read('backup')
    fixture.send('actor_private', {'kind': 'result', 'value': '10:30'})
    assert not evaluate(case, fixture)['passed']
    with pytest.raises(ValueError, match='case_fixture_mismatch'):
        evaluate(replace(case, name='missing'), fixture)
