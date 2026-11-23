"""Unwired policy for evidence-based work checkpoints and safe retry decisions.

This module does not dispatch work, call Jev, alter a budget or authenticate a
user. A future host must supply current claim-bound facts and an authenticated,
task-scoped continuation grant, persist the decision, and independently enforce
hard limits, cancellation, authorization and effect verification.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Literal


Metric = Literal['actions', 'model_calls', 'elapsed_ms', 'input_tokens', 'output_tokens']
_METRICS = {'actions', 'model_calls', 'elapsed_ms', 'input_tokens', 'output_tokens'}


class Outcome(str, Enum):
    PROCEED = 'proceed'
    CONTINUE_WINDOW = 'continue_window'
    ASK_USER = 'ask_user'
    REVIEW = 'review'
    REPLAN = 'replan'
    BLOCKED = 'blocked'
    CANCELLED = 'cancelled'
    RECONCILE = 'reconcile'
    HARD_STOP = 'hard_stop'


class Judgment(str, Enum):
    YES = 'yes'
    NO = 'no'
    UNKNOWN = 'unknown'


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'progress_checkpoint_invalid:{name}')
    return value


def _tri(value, name):
    if value is not None and type(value) is not bool:
        raise ValueError(f'progress_checkpoint_invalid:{name}')
    return value


@dataclass(frozen=True)
class CheckpointConfig:
    """Explicit operator configuration; no default metric or thresholds."""
    metric: Metric
    interval: int
    no_threshold: float
    yes_threshold: float

    def __post_init__(self):
        if not isinstance(self.metric, str) or self.metric not in _METRICS or _count(self.interval, 'interval') == 0:
            raise ValueError('progress_checkpoint_invalid:metric_or_interval')
        values = (self.no_threshold, self.yes_threshold)
        if (any(type(x) not in (int, float) or not isfinite(x) for x in values)
                or not 0 <= self.no_threshold < self.yes_threshold <= 1):
            raise ValueError('progress_checkpoint_invalid:thresholds')


@dataclass(frozen=True)
class ContinuationGrant:
    """A host-authenticated grant scoped to one task, metric and checkpoint.

    Constructing this object is not authentication; callers must verify the
    grant against the user's stored decision and apply revocation separately.
    ``through`` is a strict upper bound on the counter, not a new hard limit.
    """
    task_id: str
    metric: Metric
    checkpoint: int
    through: int

    def __post_init__(self):
        if (not isinstance(self.task_id, str) or not self.task_id.strip()
                or not isinstance(self.metric, str) or self.metric not in _METRICS
                or _count(self.through, 'through') <= _count(self.checkpoint, 'checkpoint')):
            raise ValueError('progress_checkpoint_invalid:grant')


@dataclass(frozen=True)
class CheckpointFacts:
    task_id: str
    counter: int
    last_checkpoint: int
    cancelled: bool | None
    uncertain_effect: bool | None
    hard_limit_reached: bool | None
    # A trusted host reports a concrete blocker and whether another known,
    # authorized route remains. False is not a proof about unknown routes.
    blocker_confirmed: bool | None
    alternative_available: bool | None
    # A bounded model judgment over relevant observed action history. It is
    # neither an authority grant nor proof of an effect or task completion.
    progress_probability: float | None

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError('progress_checkpoint_invalid:task_id')
        if _count(self.last_checkpoint, 'last_checkpoint') > _count(self.counter, 'counter'):
            raise ValueError('progress_checkpoint_invalid:counter_order')
        for name in ('cancelled', 'uncertain_effect', 'hard_limit_reached', 'blocker_confirmed', 'alternative_available'):
            _tri(getattr(self, name), name)
        p = self.progress_probability
        if p is not None and (type(p) not in (int, float) or not isfinite(p) or not 0 <= p <= 1):
            raise ValueError('progress_checkpoint_invalid:probability')


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    reason: str
    checkpoint_due: bool
    progress: Judgment


def classify_progress(probability: float | None, config: CheckpointConfig) -> Judgment:
    if probability is None:
        return Judgment.UNKNOWN
    if type(probability) not in (int, float) or not isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('progress_checkpoint_invalid:probability')
    if probability >= config.yes_threshold:
        return Judgment.YES
    if probability <= config.no_threshold:
        return Judgment.NO
    return Judgment.UNKNOWN


def decide(config: CheckpointConfig, facts: CheckpointFacts, grant: ContinuationGrant | None = None) -> Decision:
    """Decide a handoff from facts; never execute, retry or infer completion."""
    if not isinstance(config, CheckpointConfig) or not isinstance(facts, CheckpointFacts):
        raise ValueError('progress_checkpoint_invalid:input')
    if grant is not None and not isinstance(grant, ContinuationGrant):
        raise ValueError('progress_checkpoint_invalid:grant')
    due = facts.counter - facts.last_checkpoint >= config.interval
    progress = classify_progress(facts.progress_probability, config)
    def result(outcome, reason):
        return Decision(outcome, reason, due, progress)
    # These guards apply on every invocation, including between checkpoints.
    if facts.cancelled is True:
        return result(Outcome.CANCELLED, 'cancelled')
    if facts.uncertain_effect is True:
        return result(Outcome.RECONCILE, 'effect_unconfirmed')
    if facts.hard_limit_reached is True:
        return result(Outcome.HARD_STOP, 'hard_limit_reached')
    if facts.cancelled is None or facts.uncertain_effect is None or facts.hard_limit_reached is None:
        return result(Outcome.REVIEW, 'required_fact_unknown')
    if facts.blocker_confirmed is None:
        return result(Outcome.REVIEW, 'blocker_status_unknown')
    # A known failed route should not be retried merely to consume the budget.
    # An alternate route must still be separately authorized by the host.
    if facts.blocker_confirmed:
        if facts.alternative_available is None:
            return result(Outcome.REVIEW, 'alternative_unknown')
        if not facts.alternative_available:
            return result(Outcome.BLOCKED, 'no_known_authorized_alternative')
        if not due:
            return result(Outcome.REPLAN, 'known_alternative')
    if not due:
        return result(Outcome.PROCEED, 'checkpoint_not_due')
    if progress is Judgment.UNKNOWN:
        return result(Outcome.REVIEW, 'progress_unknown')
    if progress is Judgment.NO:
        return result(Outcome.ASK_USER, 'no_observed_progress')
    valid_grant = (grant is not None and grant.task_id == facts.task_id and grant.metric == config.metric
                   and grant.checkpoint == facts.counter and grant.through > facts.counter)
    if not valid_grant:
        return result(Outcome.ASK_USER, 'continuation_not_authorized')
    return result(Outcome.REPLAN if facts.blocker_confirmed else Outcome.CONTINUE_WINDOW,
                  'authorized_alternative' if facts.blocker_confirmed else 'authorized_progress')


class RetryOutcome(str, Enum):
    ALLOWED = 'allowed'
    WAIT = 'wait'
    DENIED = 'denied'
    REVIEW = 'review'


def decide_retry(*, repeat_safe: bool | None, retry_authorized: bool | None,
                 effect_uncertain: bool | None, cancelled: bool | None,
                 now_ms: int, last_attempt_ms: int, not_before_ms: int | None,
                 attempts_used: int, attempt_limit: int) -> RetryOutcome:
    """A retry is a separate deterministic gate, not a checkpoint decision.

    The caller must establish operation identity, bounded attempts, safe retry
    semantics, and an explicit backoff deadline. No model score establishes
    idempotency or permission; unavailable facts never permit dispatch.
    """
    for name, value in (('repeat_safe', repeat_safe), ('retry_authorized', retry_authorized),
                        ('effect_uncertain', effect_uncertain), ('cancelled', cancelled)):
        _tri(value, name)
    _count(now_ms, 'now_ms')
    _count(last_attempt_ms, 'last_attempt_ms')
    _count(attempts_used, 'attempts_used')
    _count(attempt_limit, 'attempt_limit')
    if not_before_ms is not None:
        _count(not_before_ms, 'not_before_ms')
    if (cancelled is True or effect_uncertain is True or repeat_safe is False or retry_authorized is False
            or attempts_used >= attempt_limit or now_ms < last_attempt_ms
            or (not_before_ms is not None and not_before_ms <= last_attempt_ms)):
        return RetryOutcome.DENIED
    if (cancelled is None or effect_uncertain is None or repeat_safe is None
            or retry_authorized is None or not_before_ms is None):
        return RetryOutcome.REVIEW
    return RetryOutcome.WAIT if now_ms < not_before_ms else RetryOutcome.ALLOWED
