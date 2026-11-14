"""Claim-bound model reservation for the isolated candidate experiment.

Not wired to a daemon or a provider. A reservation does not authorize or
dispatch a call. Production processing still requires fenced publication,
recovery, and a mixed-writer protocol.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
import sqlite3
from typing import Callable

from .claim_fence import ClaimFenceUnavailable, ClaimReceipt, _time, fenced_claim_transaction
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import (
    Checkpoint, CheckpointConflict, ExperimentalCheckpointStore,
)
from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetExhausted


def reserve_model_under_claim(
    store: ExperimentalCheckpointStore, claim: ClaimReceipt, *, task_id: str,
    actor: str, expected_version: int, reservation_id: str,
    input_tokens: int, output_tokens: int, clock: Callable[[], datetime],
) -> Checkpoint:
    """Fence a model reservation against the current direct-owner claim.

    Only the existing shared file-backed candidate checkpoint is supported.
    This operation has no external I/O, cannot settle or release a marker, and
    does not persist the claim generation or fence later settlement, and does
    not establish authorization. A cancelled or superseded checkpoint,
    unresolved usage, stale claim, clock failure, or cross-store identity
    fails closed. The clock is trusted; unobserved wall-clock regression cannot
    be detected across calls or restarts.
    """
    if (not isinstance(store, ExperimentalCheckpointStore) or not isinstance(claim, ClaimReceipt)
            or not isinstance(task_id, str) or not task_id.strip()
            or not isinstance(actor, str) or not actor.strip()
            or not isinstance(reservation_id, str) or not reservation_id.strip()
            or type(expected_version) is not int or expected_version < 1
            or type(input_tokens) is not int or input_tokens <= 0
            or type(output_tokens) is not int or output_tokens <= 0 or not callable(clock)):
        raise ValueError('claim_model_reservation_arguments_invalid')
    if claim.user_id != actor:
        raise ClaimFenceUnavailable('claim_model_reservation_unavailable')

    # Guard every injected time sample, including the helper's post-write
    # sample; a broken clock cannot authorize a transaction commit.
    previous = None
    def sample():
        nonlocal previous
        try:
            now = _time(clock())
        except Exception:
            raise ClaimFenceUnavailable('claim_model_reservation_clock_unavailable') from None
        if previous is not None and now < previous:
            raise ClaimFenceUnavailable('claim_model_reservation_clock_reversed')
        previous = now
        return now

    rejection = None
    result = None
    with store.store._connect() as conn:
        try:
            main = next((row[2] for row in conn.execute('PRAGMA database_list') if row[1] == 'main'), '')
            if not main or not os.path.samefile(main, store.store.db_path):
                raise ClaimFenceUnavailable('claim_model_reservation_store_unavailable')
            with fenced_claim_transaction(conn, claim, clock=sample):
                if conn.execute('SELECT 1 FROM main.v2_users WHERE user_id=? AND is_active=1', (actor,)).fetchone() is None:
                    raise ClaimFenceUnavailable('claim_model_reservation_unavailable')
                row = store._row(conn, task_id)
                raw = store._validate_identity(row, task_id, actor, expected_version)
                if raw.get('message_id') != claim.message_id:
                    raise ClaimFenceUnavailable('claim_model_reservation_unavailable')
                original = dict(conn.execute('SELECT * FROM main.v2_task_checkpoints WHERE task_id=?',
                                             (task_id,)).fetchone())
                state, budget = store._decode(raw, task_id, actor)
                if state.inflight:
                    raise CheckpointConflict('single_loop_inflight_requires_reconciliation')
                if budget.usage()['unknown_usage']:
                    raise CheckpointConflict('single_loop_model_usage_unresolved')
                if any(item['kind'] == 'model' and item['status'] == 'settled'
                       and any(item['actual'][key] > item['reserved'][key] for key in ('input_tokens', 'output_tokens'))
                       for item in budget.to_dict()['reservations'].values()):
                    raise CheckpointConflict('single_loop_model_reservation_overrun')
                original_revision = budget.revision
                try:
                    budget.reserve(reservation_id, kind='model', now_ms=int(sample().timestamp() * 1000),
                                   amounts={'actions': 0, 'model_calls': 1,
                                            'input_tokens': input_tokens, 'output_tokens': output_tokens})
                except BudgetExhausted as exc:
                    rejection = exc
                    if budget.revision != original_revision:
                        result = store._write(conn, row, raw, state, budget)
                else:
                    state.inflight = reservation_id
                    result = store._write(conn, row, raw, state, budget)
                if result is not None:
                    # Reject unexpected checkpoint-trigger changes within the
                    # transaction rather than persisting a different action.
                    saved = store._row(conn, task_id)
                    checked = store._validate_identity(saved, task_id, actor, result.version)
                    saved_state, saved_budget = store._decode(checked, task_id, actor)
                    if (checked != raw or saved_state.snapshot() != state.snapshot()
                            or saved_budget.to_dict() != budget.to_dict()):
                        raise ClaimFenceUnavailable('claim_model_reservation_changed')
                    after = dict(conn.execute('SELECT * FROM main.v2_task_checkpoints WHERE task_id=?',
                                              (task_id,)).fetchone())
                    for field in ('version', 'task_state_json', 'updated_at'):
                        original.pop(field, None)
                        after.pop(field, None)
                    if after != original:
                        raise ClaimFenceUnavailable('claim_model_reservation_changed')
                if conn.execute('SELECT 1 FROM main.v2_users WHERE user_id=? AND is_active=1', (actor,)).fetchone() is None:
                    raise ClaimFenceUnavailable('claim_model_reservation_unavailable')
        except (OSError, sqlite3.Error, StopIteration):
            raise ClaimFenceUnavailable('claim_model_reservation_storage_unavailable') from None
    if rejection is not None:
        raise rejection
    assert result is not None
    return result
