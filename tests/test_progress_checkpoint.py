from dataclasses import replace

import pytest

from alphonse.agent_v2.core.intelligence.simple_loop.progress_checkpoint import (
    CheckpointConfig, CheckpointFacts, ContinuationGrant, Judgment, Outcome,
    RetryOutcome, classify_progress, decide, decide_retry,
)


@pytest.fixture
def config():
    return CheckpointConfig('actions', 10, 0.2, 0.8)


@pytest.fixture
def facts():
    return CheckpointFacts('task', 10, 0, False, False, False, False, None, 0.9)


def test_configuration_and_state_are_explicit_and_validated(config, facts):
    with pytest.raises(TypeError):
        CheckpointConfig()
    for kwargs in ({'interval': 0}, {'metric': []}, {'no_threshold': 0.9}, {'yes_threshold': float('nan')}):
        with pytest.raises(ValueError):
            replace(config, **kwargs)
    for kwargs in ({'counter': True}, {'last_checkpoint': 11}, {'cancelled': 0}, {'progress_probability': float('nan')}):
        with pytest.raises(ValueError):
            replace(facts, **kwargs)


def test_calibrated_judgment_keeps_unknown_distinct(config):
    assert classify_progress(None, config) is Judgment.UNKNOWN
    assert classify_progress(0.5, config) is Judgment.UNKNOWN
    assert classify_progress(0.2, config) is Judgment.NO
    assert classify_progress(0.8, config) is Judgment.YES


def test_checkpoint_requires_scoped_user_grant(config, facts):
    assert decide(config, replace(facts, counter=9)).outcome is Outcome.PROCEED
    assert decide(config, facts).outcome is Outcome.ASK_USER
    grant = ContinuationGrant('task', 'actions', 10, 15)
    assert decide(config, facts, grant).outcome is Outcome.CONTINUE_WINDOW
    assert decide(config, facts, replace(grant, task_id='another')).outcome is Outcome.ASK_USER
    assert decide(config, facts, replace(grant, metric='model_calls')).outcome is Outcome.ASK_USER
    assert decide(config, replace(facts, counter=11), grant).outcome is Outcome.ASK_USER
    with pytest.raises(ValueError):
        replace(grant, through=10)


@pytest.mark.parametrize(('probability', 'outcome'), [
    (None, Outcome.REVIEW), (0.5, Outcome.REVIEW), (0.1, Outcome.ASK_USER),
])
def test_grant_does_not_turn_uncertain_or_absent_progress_into_progress(config, facts, probability, outcome):
    grant = ContinuationGrant('task', 'actions', 10, 15)
    assert decide(config, replace(facts, progress_probability=probability), grant).outcome is outcome


@pytest.mark.parametrize(('field', 'value', 'outcome'), [
    ('cancelled', True, Outcome.CANCELLED), ('uncertain_effect', True, Outcome.RECONCILE),
    ('hard_limit_reached', True, Outcome.HARD_STOP), ('cancelled', None, Outcome.REVIEW),
    ('blocker_confirmed', None, Outcome.REVIEW),
])
def test_guards_override_continuation_even_between_checkpoints(config, facts, field, value, outcome):
    grant = ContinuationGrant('task', 'actions', 10, 15)
    assert decide(config, replace(facts, **{field: value}), grant).outcome is outcome
    assert decide(config, replace(facts, counter=1, **{field: value})).outcome is outcome


def test_blocker_does_not_trigger_repeated_attempts(config, facts):
    blocked = replace(facts, counter=1, blocker_confirmed=True, alternative_available=False)
    assert decide(config, blocked).outcome is Outcome.BLOCKED
    assert decide(config, replace(blocked, alternative_available=None)).outcome is Outcome.REVIEW
    assert decide(config, replace(blocked, alternative_available=True)).outcome is Outcome.REPLAN
    assert decide(config, replace(facts, blocker_confirmed=True, alternative_available=True)).outcome is Outcome.ASK_USER


def test_retry_requires_explicit_safe_bounded_authorization_and_backoff():
    args = dict(repeat_safe=True, retry_authorized=True, effect_uncertain=False,
                cancelled=False, now_ms=200, last_attempt_ms=100, not_before_ms=300,
                attempts_used=1, attempt_limit=3)
    assert decide_retry(**args) is RetryOutcome.WAIT
    assert decide_retry(**(args | {'now_ms': 300})) is RetryOutcome.ALLOWED
    for change in ({'repeat_safe': False}, {'retry_authorized': False}, {'effect_uncertain': True},
                   {'cancelled': True}, {'attempts_used': 3}, {'not_before_ms': 100}, {'now_ms': 99}):
        assert decide_retry(**(args | change)) is RetryOutcome.DENIED
    for change in ({'repeat_safe': None}, {'retry_authorized': None}, {'not_before_ms': None}):
        assert decide_retry(**(args | change)) is RetryOutcome.REVIEW
