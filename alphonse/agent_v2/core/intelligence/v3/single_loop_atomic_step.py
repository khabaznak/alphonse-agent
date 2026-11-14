"""Synthetic, model-only use of the experimental task checkpoint transaction.

This is not an engine, dispatcher, provider adapter, or production entry point.
It reserves and settles one injected model decision against the existing task
row. It neither dispatches a selected tool nor accepts an answer/question as a
completed task. A real host must still enforce actor-scoped authorization,
provider limits, cancellation and atomic effect/evidence reconciliation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import sqlite3
from typing import Callable
from uuid import uuid4

from .single_loop_candidate import (
    CandidateHost, CandidateState, _cancelled, _is_fresh, _project, _projection_fingerprint,
)
from .single_loop_checkpoint import CheckpointConflict, ExperimentalCheckpointStore
from .task_budget import BudgetExhausted
from alphonse.agent_v2.core.messages.claim_fence import (
    ClaimFenceUnavailable, ClaimReceipt, ClaimSuperseded, _time, fenced_claim_transaction,
)
from alphonse.agent_v2.core.messages.experimental_claim_checkpoint import (
    MODEL_CLAIM_KEY, _claim_digest, reserve_model_under_claim, settle_model_under_claim,
)


@dataclass(frozen=True)
class AtomicStepResult:
    status: str
    reason: str
    checkpoint_version: int | None


def run_atomic_step(state: CandidateState, host: CandidateHost, checkpoints: ExperimentalCheckpointStore,
                    *, expected_version: int, claim: ClaimReceipt | None = None,
                    claim_clock: Callable[[], datetime] | None = None) -> AtomicStepResult:
    """Run one synthetic decision, leaving every effectful path disabled.

    The only permitted external operation is the host's injected chooser;
    callers are responsible for using a synthetic chooser. All budget and
    in-flight transitions are durable before/after that call. Cancellation after
    dispatch or an interrupted process leaves the model marker in place. Any
    tool decision is withheld because tool evidence cannot yet be settled.
    An explicit claim and clock opt into claim-fenced model accounting. This
    synthetic adapter remains unwired: no daemon, tool, user output or external
    provider execution is authorized by the claim.
    """
    if (claim is None) != (claim_clock is None) or (claim is not None and
            (not isinstance(claim, ClaimReceipt) or not callable(claim_clock))):
        return AtomicStepResult('blocked', 'claim_configuration_invalid', None)
    previous = None
    def checked_clock():
        nonlocal previous
        try:
            now = _time(claim_clock())
        except Exception:
            raise ClaimFenceUnavailable('claim_clock_unavailable') from None
        if previous is not None and now < previous:
            raise ClaimFenceUnavailable('claim_clock_reversed')
        previous = now
        return now
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
        if claim is not None:
            # The projection can read actor-scoped sources. Check the exact
            # queue claim and checkpoint before invoking any source callbacks;
            # a reclaimed worker must not read under a known-stale receipt.
            # This transaction cannot make later reads atomic with claim loss.
            with checkpoints.store._connect() as conn:
                with fenced_claim_transaction(conn, claim, clock=checked_clock):
                    if (claim.user_id != state.actor
                            or conn.execute('SELECT 1 FROM main.v2_users WHERE user_id=? AND is_active=1',
                                            (state.actor,)).fetchone() is None):
                        raise ClaimFenceUnavailable('claim_actor_unavailable')
                    row = checkpoints._row(conn, state.task_id)
                    raw = checkpoints._validate_identity(row, state.task_id, state.actor, checkpoint.version)
                    current_state, _ = checkpoints._decode(raw, state.task_id, state.actor)
                    if (raw.get('message_id') != claim.message_id
                            or current_state.snapshot() != state.snapshot()):
                        raise ClaimFenceUnavailable('claim_checkpoint_unavailable')
        projection = _project(host, state)
        fingerprint = _projection_fingerprint(projection)
        if (type(host.input_reservation) is not int or host.input_reservation <= 0
                or type(host.output_reservation) is not int or host.output_reservation <= 0
                or len(projection.text.encode('utf-8')) > host.input_reservation):
            return AtomicStepResult('blocked', 'input_reservation_invalid_or_too_small', checkpoint.version)
        reservation_id = 'model-' + uuid4().hex
        if claim is None:
            reserved = checkpoints.reserve(state.task_id, state.actor, expected_version=checkpoint.version,
                reservation_id=reservation_id, kind='model', now_ms=host.now_ms(), amounts={
                    'actions': 0, 'model_calls': 1, 'input_tokens': host.input_reservation,
                    'output_tokens': host.output_reservation})
        else:
            reserved = reserve_model_under_claim(checkpoints, claim, task_id=state.task_id,
                actor=state.actor, expected_version=checkpoint.version, reservation_id=reservation_id,
                input_tokens=host.input_reservation, output_tokens=host.output_reservation, clock=checked_clock)
    except (BudgetExhausted, CheckpointConflict, ClaimFenceUnavailable, ClaimSuperseded, sqlite3.Error,
            ValueError, KeyError, PermissionError):
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
        # Projection freshness checks and checkpoint I/O can consume the
        # remaining work window after the reservation. Do not dispatch with an
        # expired or regressed clock. Retain the reservation conservatively:
        # this adapter has no durable proof-of-non-dispatch release protocol.
        try:
            now = host.now_ms()
        except Exception:
            return AtomicStepResult('blocked', 'deadline_or_clock_invalid_before_model', current.version)
        budget = current.budget
        if (type(now) is not int or now < budget.last_observed_at_ms
                or now - budget.started_at_ms >= budget.limits.elapsed_ms - budget.limits.final_reserve_ms):
            return AtomicStepResult('blocked', 'deadline_or_clock_invalid_before_model', current.version)
        if claim is not None:
            # Check the same actor-bound task and claim under one writer lock
            # immediately before the injected model call. The later external
            # call cannot be atomic with this check; on loss, settlement blocks.
            with checkpoints.store._connect() as conn:
                with fenced_claim_transaction(conn, claim, clock=checked_clock):
                    lease = conn.execute('SELECT lease_expires_at FROM main.v2_inbound_messages WHERE message_id=?',
                                         (claim.message_id,)).fetchone()
                    if lease is None:
                        raise ClaimFenceUnavailable('claim_lease_unavailable')
                    try:
                        lease_expires = _time(datetime.fromisoformat(lease[0]))
                    except (TypeError, ValueError, ClaimFenceUnavailable):
                        raise ClaimFenceUnavailable('claim_lease_unavailable') from None
                    if conn.execute('SELECT 1 FROM main.v2_users WHERE user_id=? AND is_active=1',
                                    (state.actor,)).fetchone() is None:
                        raise ClaimFenceUnavailable('claim_actor_unavailable')
                    row = checkpoints._row(conn, state.task_id)
                    raw = checkpoints._validate_identity(row, state.task_id, state.actor, reserved.version)
                    current_state, _ = checkpoints._decode(raw, state.task_id, state.actor)
                    marker = raw['metadata'].get(MODEL_CLAIM_KEY)
                    if (raw.get('message_id') != claim.message_id or current_state.inflight != reservation_id
                            or not isinstance(marker, dict) or type(marker.get('schema_version')) is not int
                            or marker != {'schema_version': 1, 'reservation_id': reservation_id,
                                          'claim_digest': _claim_digest(claim)}):
                        raise ClaimFenceUnavailable('claim_model_binding_unavailable')
            if _cancelled(host):
                return AtomicStepResult('cancelled', 'cancelled_before_dispatch', reserved.version)
            try:
                latest = host.now_ms()
            except Exception:
                return AtomicStepResult('blocked', 'deadline_or_clock_invalid_before_model', reserved.version)
            if (type(latest) is not int or latest < now
                    or latest - budget.started_at_ms >= budget.limits.elapsed_ms - budget.limits.final_reserve_ms):
                return AtomicStepResult('blocked', 'deadline_or_clock_invalid_before_model', reserved.version)
            claim_now = checked_clock()
            claim_now_ms = int(claim_now.timestamp() * 1000)
            if (claim_now_ms < budget.last_observed_at_ms
                    or claim_now >= lease_expires
                    or claim_now_ms - budget.started_at_ms >= budget.limits.elapsed_ms - budget.limits.final_reserve_ms):
                return AtomicStepResult('blocked', 'deadline_or_clock_invalid_before_model', reserved.version)
    except (CheckpointConflict, ClaimFenceUnavailable, ClaimSuperseded, sqlite3.Error, ValueError, KeyError, TypeError):
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
        if claim is None:
            settled = checkpoints.settle(state.task_id, state.actor, expected_version=reserved.version,
                reservation_id=reservation_id, outcome='success' if isinstance(decision, dict) else 'failure',
                input_tokens=usage.get('input_tokens'), output_tokens=usage.get('output_tokens'))
        else:
            settled = settle_model_under_claim(checkpoints, claim, task_id=state.task_id, actor=state.actor,
                expected_version=reserved.version, reservation_id=reservation_id,
                outcome='success' if isinstance(decision, dict) else 'failure',
                input_tokens=usage.get('input_tokens'), output_tokens=usage.get('output_tokens'), clock=checked_clock)
    except (CheckpointConflict, ClaimFenceUnavailable, ClaimSuperseded, ValueError, KeyError):
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
