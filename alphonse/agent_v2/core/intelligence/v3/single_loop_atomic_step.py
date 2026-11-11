"""Synthetic, model-only use of the experimental task checkpoint transaction.

This is not an engine, dispatcher, provider adapter, or production entry point.
It reserves and settles one injected model decision against the existing task
row. It neither dispatches a selected tool nor accepts an answer/question as a
completed task. A real host must still enforce actor-scoped authorization,
provider limits, cancellation and atomic effect/evidence reconciliation.
"""
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from .single_loop_candidate import (
    CandidateHost, CandidateState, _cancelled, _is_fresh, _project, _projection_fingerprint,
)
from .single_loop_checkpoint import CheckpointConflict, ExperimentalCheckpointStore
from .task_budget import BudgetExhausted


@dataclass(frozen=True)
class AtomicStepResult:
    status: str
    reason: str
    checkpoint_version: int | None


def run_atomic_step(state: CandidateState, host: CandidateHost, checkpoints: ExperimentalCheckpointStore,
                    *, expected_version: int) -> AtomicStepResult:
    """Run one synthetic decision, leaving every effectful path disabled.

    The only permitted external operation is the host's injected chooser;
    callers are responsible for using a synthetic chooser. All budget and
    in-flight transitions are durable before/after that call. Cancellation after
    dispatch or an interrupted process leaves the model marker in place. Any
    tool decision is withheld because tool evidence cannot yet be settled.
    """
    try:
        checkpoint = checkpoints.load(state.task_id, state.actor, expected_version=expected_version)
        if (checkpoint.state.snapshot() != state.snapshot()
                or state.task_id != getattr(host.task, 'task_id', None)
                or state.actor != getattr(host.task, 'user', None)):
            return AtomicStepResult('blocked', 'checkpoint_or_actor_mismatch', checkpoint.version)
        if state.inflight:
            return AtomicStepResult('blocked', 'inflight_requires_reconciliation', checkpoint.version)
        if _cancelled(host):
            return AtomicStepResult('cancelled', 'cancelled_before_dispatch', checkpoint.version)
        projection = _project(host, state)
        fingerprint = _projection_fingerprint(projection)
        if (type(host.input_reservation) is not int or host.input_reservation <= 0
                or type(host.output_reservation) is not int or host.output_reservation <= 0
                or len(projection.text.encode('utf-8')) > host.input_reservation):
            return AtomicStepResult('blocked', 'input_reservation_invalid_or_too_small', checkpoint.version)
        reservation_id = 'model-' + uuid4().hex
        reserved = checkpoints.reserve(state.task_id, state.actor, expected_version=checkpoint.version,
            reservation_id=reservation_id, kind='model', now_ms=host.now_ms(), amounts={
                'actions': 0, 'model_calls': 1, 'input_tokens': host.input_reservation,
                'output_tokens': host.output_reservation})
    except (BudgetExhausted, CheckpointConflict, ValueError, KeyError, PermissionError):
        # A rejected budget observation may have advanced the row version.
        return AtomicStepResult('blocked', 'checkpoint_projection_or_budget_unavailable', None)

    if _cancelled(host) or not _is_fresh(host, state, fingerprint):
        return AtomicStepResult('blocked', 'cancelled_or_evidence_changed_before_model', reserved.version)
    try:
        # Another writer may have cancelled the durable task while an injected
        # cancellation callback was stale. Recheck immediately before model I/O.
        # This narrows, but cannot eliminate, the external-dispatch race.
        current = checkpoints.load(state.task_id, state.actor, expected_version=reserved.version)
        if current.state.inflight != reservation_id:
            return AtomicStepResult('blocked', 'inflight_changed_before_model', current.version)
    except (CheckpointConflict, ValueError, KeyError):
        return AtomicStepResult('blocked', 'checkpoint_changed_before_model', None)
    try:
        decision = host.choose(projection.text, host.output_reservation)
    except Exception:
        decision = None
    if _cancelled(host):
        return AtomicStepResult('cancelled', 'cancelled_after_model', reserved.version)
    usage = decision.get('usage') if isinstance(decision, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    try:
        settled = checkpoints.settle(state.task_id, state.actor, expected_version=reserved.version,
            reservation_id=reservation_id, outcome='success' if isinstance(decision, dict) else 'failure',
            input_tokens=usage.get('input_tokens'), output_tokens=usage.get('output_tokens'))
    except (CheckpointConflict, ValueError, KeyError):
        return AtomicStepResult('blocked', 'model_settlement_unavailable', None)
    if (type(usage.get('input_tokens')) is not int or type(usage.get('output_tokens')) is not int
            or usage['input_tokens'] > host.input_reservation or usage['output_tokens'] > host.output_reservation):
        return AtomicStepResult('blocked', 'model_usage_unknown_or_overrun', settled.version)
    if not _is_fresh(host, state, fingerprint):
        return AtomicStepResult('blocked', 'evidence_changed_or_revoked', settled.version)
    if not isinstance(decision, dict):
        return AtomicStepResult('blocked', 'model_unavailable', settled.version)
    if decision.get('kind') == 'tool':
        return AtomicStepResult('blocked', 'tool_dispatch_and_evidence_not_integrated', settled.version)
    if decision.get('kind') in {'answer', 'question'}:
        return AtomicStepResult('blocked', 'communication_and_completion_not_integrated', settled.version)
    return AtomicStepResult('blocked', 'invalid_model_decision', settled.version)
