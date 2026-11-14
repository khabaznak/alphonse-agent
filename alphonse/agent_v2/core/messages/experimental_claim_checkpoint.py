"""Claim-bound model reservation for the isolated candidate experiment.

Not wired to a daemon or a provider. A reservation does not authorize or
dispatch a call. Production processing still requires fenced publication,
recovery, and a mixed-writer protocol.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import hmac
import json
import os
import re
import sqlite3
from typing import Callable

from .claim_fence import ClaimFenceUnavailable, ClaimReceipt, _time, fenced_claim_transaction
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import (
    Checkpoint, CheckpointConflict, ExperimentalCheckpointStore,
)
from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetExhausted

MODEL_CLAIM_KEY = 'single_loop_model_claim'


def _claim_digest(claim: ClaimReceipt) -> str:
    """Bind one opaque acquisition without persisting the raw generation."""
    value = ['single-loop-model-claim-v1', claim.message_id, claim.user_id,
             claim.lease_owner, claim.claim_generation]
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()


def reserve_model_under_claim(
    store: ExperimentalCheckpointStore, claim: ClaimReceipt, *, task_id: str,
    actor: str, expected_version: int, reservation_id: str,
    input_tokens: int, output_tokens: int, clock: Callable[[], datetime],
) -> Checkpoint:
    """Fence a model reservation against the current direct-owner claim.

    Only the existing shared file-backed candidate checkpoint is supported.
    This operation has no external I/O, cannot settle or release a marker, and
    records a digest binding the reservation to this exact acquisition. It does
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
                if MODEL_CLAIM_KEY in raw['metadata']:
                    raise CheckpointConflict('claim_model_binding_unresolved')
                if 'fixture_selected_answer' in raw['metadata']:
                    raise CheckpointConflict('fixture_selected_answer_unresolved')
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
                    raw['metadata'][MODEL_CLAIM_KEY] = {
                        'schema_version': 1, 'reservation_id': reservation_id,
                        'claim_digest': _claim_digest(claim),
                    }
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


def settle_model_under_claim(
    store: ExperimentalCheckpointStore, claim: ClaimReceipt, *, task_id: str,
    actor: str, expected_version: int, reservation_id: str, outcome: str,
    input_tokens: int | None, output_tokens: int | None,
    clock: Callable[[], datetime], fixture_answer: str | None = None, fixture_mode: bool = False,
) -> Checkpoint:
    """Settle one model reservation only under the exact acquiring claim.

    This is not a tool/effect settlement, authorization, or proof of delivery.
    A lost claim, cancellation, uncertain worker ownership, or mismatched
    reservation keeps the full in-flight marker for future reconciliation.
    The clock is trusted; unobserved wall-clock regression is not detected.
    """
    if (not isinstance(store, ExperimentalCheckpointStore) or not isinstance(claim, ClaimReceipt)
            or not isinstance(task_id, str) or not task_id.strip()
            or not isinstance(actor, str) or not actor.strip()
            or not isinstance(reservation_id, str) or not reservation_id.strip()
            or type(expected_version) is not int or expected_version < 1
            or outcome not in {'success', 'failure', 'timeout'} or not callable(clock)
            or type(fixture_mode) is not bool
            or (fixture_answer is not None and (not fixture_mode or outcome != 'success' or not isinstance(fixture_answer, str)
                or not fixture_answer.strip() or len(fixture_answer.encode('utf-8')) > 12000))):
        raise ValueError('claim_model_settlement_arguments_invalid')
    if claim.user_id != actor:
        raise ClaimFenceUnavailable('claim_model_settlement_unavailable')

    previous = None
    def sample():
        nonlocal previous
        try:
            now = _time(clock())
        except Exception:
            raise ClaimFenceUnavailable('claim_model_settlement_clock_unavailable') from None
        if previous is not None and now < previous:
            raise ClaimFenceUnavailable('claim_model_settlement_clock_reversed')
        previous = now
        return now

    with store.store._connect() as conn:
        try:
            main = next((row[2] for row in conn.execute('PRAGMA database_list') if row[1] == 'main'), '')
            if not main or not os.path.samefile(main, store.store.db_path):
                raise ClaimFenceUnavailable('claim_model_settlement_store_unavailable')
            with fenced_claim_transaction(conn, claim, clock=sample):
                if conn.execute('SELECT 1 FROM main.v2_users WHERE user_id=? AND is_active=1', (actor,)).fetchone() is None:
                    raise ClaimFenceUnavailable('claim_model_settlement_unavailable')
                row = store._row(conn, task_id)
                raw = store._validate_identity(row, task_id, actor, expected_version)
                if raw.get('message_id') != claim.message_id:
                    raise ClaimFenceUnavailable('claim_model_settlement_unavailable')
                original = dict(conn.execute('SELECT * FROM main.v2_task_checkpoints WHERE task_id=?',
                                             (task_id,)).fetchone())
                state, budget = store._decode(raw, task_id, actor)
                bound = raw['metadata'].get(MODEL_CLAIM_KEY)
                if (not isinstance(bound, dict) or set(bound) != {'schema_version', 'reservation_id', 'claim_digest'}
                        or type(bound['schema_version']) is not int or bound['schema_version'] != 1
                        or bound['reservation_id'] != reservation_id
                        or not isinstance(bound['claim_digest'], str)
                        or re.fullmatch(r'[0-9a-f]{64}', bound['claim_digest']) is None
                        or not hmac.compare_digest(bound['claim_digest'], _claim_digest(claim))
                        or state.inflight != reservation_id):
                    raise CheckpointConflict('claim_model_binding_unavailable')
                pending = budget.to_dict()['reservations'].get(reservation_id)
                if not isinstance(pending, dict) or pending['kind'] != 'model' or pending['status'] != 'reserved':
                    raise CheckpointConflict('claim_model_binding_unavailable')
                if fixture_mode:
                    # Evaluation-only intent: exact selected bytes and acquisition
                    # are committed with model accounting. This does not establish
                    # production authority, grounding, or a provider effect.
                    if ('fixture_selected_answer' in raw['metadata'] or
                            (fixture_answer is not None and (type(input_tokens) is not int or
                            type(output_tokens) is not int or input_tokens < 0 or output_tokens < 0 or
                            input_tokens > pending['reserved']['input_tokens'] or
                            output_tokens > pending['reserved']['output_tokens']))):
                        raise CheckpointConflict('fixture_selected_answer_unavailable')
                    raw['metadata']['fixture_selected_answer'] = {
                        'schema_version': 1, 'message': fixture_answer,
                        'claim_digest': _claim_digest(claim), 'accepted': False,
                    }
                budget.settle(reservation_id, outcome=outcome, input_tokens=input_tokens, output_tokens=output_tokens)
                state.inflight = ''
                del raw['metadata'][MODEL_CLAIM_KEY]
                result = store._write(conn, row, raw, state, budget)
                saved = store._row(conn, task_id)
                checked = store._validate_identity(saved, task_id, actor, result.version)
                saved_state, saved_budget = store._decode(checked, task_id, actor)
                if (checked != raw or saved_state.snapshot() != state.snapshot()
                        or saved_budget.to_dict() != budget.to_dict()):
                    raise ClaimFenceUnavailable('claim_model_settlement_changed')
                after = dict(conn.execute('SELECT * FROM main.v2_task_checkpoints WHERE task_id=?',
                                          (task_id,)).fetchone())
                for field in ('version', 'task_state_json', 'updated_at'):
                    original.pop(field, None)
                    after.pop(field, None)
                if after != original or conn.execute('SELECT 1 FROM main.v2_users WHERE user_id=? AND is_active=1', (actor,)).fetchone() is None:
                    raise ClaimFenceUnavailable('claim_model_settlement_changed')
                return result
        except (OSError, sqlite3.Error, StopIteration):
            raise ClaimFenceUnavailable('claim_model_settlement_storage_unavailable') from None


def _accept_fixture_answer(store: ExperimentalCheckpointStore, claim: ClaimReceipt, *, task_id: str,
                           actor: str, expected_version: int, answer: str,
                           clock: Callable[[], datetime]) -> Checkpoint:
    """Mark a fixture answer accepted after the caller's final freshness check.

    Only synthetic callers may trust that check. This is not production
    authorization, grounding, or atomicity with external source revocation.
    """
    with store.store._connect() as conn:
        with fenced_claim_transaction(conn, claim, clock=clock):
            if claim.user_id != actor or conn.execute(
                    'SELECT 1 FROM main.v2_users WHERE user_id=? AND is_active=1', (actor,)).fetchone() is None:
                raise ClaimFenceUnavailable('fixture_answer_unavailable')
            row = store._row(conn, task_id)
            raw = store._validate_identity(row, task_id, actor, expected_version)
            state, budget = store._decode(raw, task_id, actor)
            expected = {'schema_version': 1, 'message': answer,
                        'claim_digest': _claim_digest(claim), 'accepted': False}
            marker = raw['metadata'].get('fixture_selected_answer')
            if (raw.get('message_id') != claim.message_id or state.inflight or
                    not isinstance(marker, dict) or type(marker.get('schema_version')) is not int or
                    type(marker.get('accepted')) is not bool or marker != expected):
                raise CheckpointConflict('fixture_answer_unavailable')
            marker['accepted'] = True
            result = store._write(conn, row, raw, state, budget)
            saved = store._validate_identity(store._row(conn, task_id), task_id, actor, result.version)
            if saved != raw:
                raise ClaimFenceUnavailable('fixture_answer_changed')
            return result
