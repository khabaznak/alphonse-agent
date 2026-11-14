"""Experimental atomic task/budget transitions in existing task checkpoints.

Not registered with any production engine. This narrow adapter accepts only an
explicit experimental engine stamp and running owner-scoped tasks. Its writes
use one transaction on the existing SQLite task row, but legacy checkpoint
writers do not use CAS, so this is not a production concurrency guarantee.
No tool, provider, question, delivery or household effect is dispatched here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .single_loop_candidate import CandidateState
from .task_budget import BudgetExhausted, TaskBudget

ENGINE = 'single_loop_experimental'
STATE_KEY = 'single_loop_checkpoint'
SCHEMA_VERSION = 1
TASK_SCHEMA_VERSION = 1


class CheckpointConflict(ValueError):
    """The task is stale, unavailable, cancelled, or owned by another actor."""


@dataclass(frozen=True)
class Checkpoint:
    version: int
    state: CandidateState
    budget: TaskBudget


class ExperimentalCheckpointStore:
    """Version-fenced transitions on a real SQLiteQuestionStore checkpoint row.

    Inflight reservations and their task markers commit together. Tool
    settlement is blocked until actor/action-bound evidence validation and
    reconciliation exist. A crash between external I/O and settlement leaves
    the inflight marker in place. The caller must enforce
    actor authorization and cancellation at the actual I/O boundary; this row
    check does not fence revocation in other stores or systems.
    """

    def __init__(self, question_store: Any) -> None:
        from alphonse.agent_v2.core.questions import SQLiteQuestionStore
        if not isinstance(question_store, SQLiteQuestionStore):
            raise ValueError('single_loop_question_store_required')
        # Avoid sharing a non-isolated in-memory connection across callers.
        if getattr(question_store, '_memory_connection', None) is not None:
            raise ValueError('single_loop_file_checkpoint_required')
        self.store = question_store

    @staticmethod
    def _validate_identity(row, task_id: str, actor: str, version: int, *, allow_cancelled: bool = False) -> dict:
        if (not isinstance(task_id, str) or not task_id.strip() or not isinstance(actor, str)
                or not actor.strip() or type(version) is not int or version < 1):
            raise ValueError('single_loop_checkpoint_arguments_invalid')
        allowed_statuses = {'cancelled'} if allow_cancelled else {'running'}
        if row is None or row['version'] != version or row['owner_id'] != actor or row['status'] not in allowed_statuses:
            raise CheckpointConflict('single_loop_checkpoint_unavailable_or_stale')
        try:
            raw = json.loads(row['task_state_json'])
        except (TypeError, ValueError):
            raise ValueError('single_loop_checkpoint_invalid') from None
        if not isinstance(raw, dict) or not isinstance(raw.get('metadata'), dict):
            raise ValueError('single_loop_checkpoint_invalid')
        metadata = raw['metadata']
        if (raw.get('task_id') != task_id or raw.get('user') != actor
                or raw.get('intelligence_engine') != ENGINE
                or metadata.get('intelligence_engine', ENGINE) != ENGINE
                or type(raw.get('intelligence_schema_version')) is not int
                or raw['intelligence_schema_version'] != TASK_SCHEMA_VERSION
                or type(metadata.get('intelligence_schema_version', TASK_SCHEMA_VERSION)) is not int
                or metadata.get('intelligence_schema_version', TASK_SCHEMA_VERSION) != TASK_SCHEMA_VERSION
                or raw.get('status') not in ({'running', 'cancelled'} if allow_cancelled else {'running'})):
            raise CheckpointConflict('single_loop_checkpoint_identity_or_engine_mismatch')
        return raw

    @staticmethod
    def _decode(raw: dict, task_id: str, actor: str) -> tuple[CandidateState, TaskBudget]:
        envelope = raw['metadata'].get(STATE_KEY)
        if not isinstance(envelope, dict) or set(envelope) != {'schema_version', 'state', 'budget'}:
            raise ValueError('single_loop_envelope_invalid')
        if type(envelope['schema_version']) is not int or envelope['schema_version'] != SCHEMA_VERSION:
            raise ValueError('single_loop_envelope_version_unsupported')
        state = CandidateState.restore(envelope['state'])
        budget = TaskBudget.from_dict(envelope['budget'])
        if state.task_id != task_id or state.actor != actor or state.status != 'running':
            raise CheckpointConflict('single_loop_state_identity_or_status_mismatch')
        pending = budget.usage()['in_flight']
        if (state.inflight and pending != [state.inflight]) or (not state.inflight and pending):
            raise ValueError('single_loop_inflight_budget_mismatch')
        return state, budget

    @staticmethod
    def _write(conn, row, raw: dict, state: CandidateState, budget: TaskBudget) -> Checkpoint:
        raw['metadata'][STATE_KEY] = {'schema_version': SCHEMA_VERSION, 'state': state.snapshot(), 'budget': budget.to_dict()}
        cursor = conn.execute('''UPDATE v2_task_checkpoints SET task_state_json=?, version=version+1, updated_at=?
            WHERE task_id=? AND owner_id=? AND status='running' AND version=?''',
            (json.dumps(raw, sort_keys=True), datetime.now(timezone.utc).isoformat(), row['task_id'], row['owner_id'], row['version']))
        if cursor.rowcount != 1:
            raise CheckpointConflict('single_loop_checkpoint_unavailable_or_stale')
        return Checkpoint(row['version'] + 1, state, budget)

    @staticmethod
    def _row(conn, task_id: str):
        return conn.execute('SELECT task_id, owner_id, status, version, task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                            (task_id,)).fetchone()

    def initialize(self, task_id: str, actor: str, *, expected_version: int,
                   state: CandidateState, budget: TaskBudget) -> Checkpoint:
        if (not isinstance(state, CandidateState) or not isinstance(budget, TaskBudget)
                or state.task_id != task_id or state.actor != actor or state.status != 'running'
                or state.inflight or state.evidence_sources or state.response or state.blocker or budget.revision != 0):
            raise ValueError('single_loop_initial_state_invalid')
        # Strict round trips reject malformed objects before a write.
        CandidateState.restore(state.snapshot())
        TaskBudget.from_dict(budget.to_dict())
        with self.store._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = self._row(conn, task_id)
            raw = self._validate_identity(row, task_id, actor, expected_version)
            if 'single_loop_model_claim' in raw['metadata']:
                raise CheckpointConflict('claim_model_binding_unresolved')
            if STATE_KEY in raw['metadata']:
                raise CheckpointConflict('single_loop_already_initialized')
            return self._write(conn, row, raw, state, budget)

    def load(self, task_id: str, actor: str, *, expected_version: int) -> Checkpoint:
        with self.store._connect() as conn:
            row = self._row(conn, task_id)
            raw = self._validate_identity(row, task_id, actor, expected_version)
            state, budget = self._decode(raw, task_id, actor)
            return Checkpoint(row['version'], state, budget)

    def inspect_cancelled(self, task_id: str, actor: str, *, expected_version: int) -> dict:
        """Read only the cancelled task's unresolved reservation, if any.

        This exposes no source content or result and cannot settle, retry, or
        resume an action. The caller must independently authenticate the actor.
        Legacy cancellation may change the row status without rewriting the
        serialized task status, so either running or cancelled JSON is accepted.
        """
        with self.store._connect() as conn:
            row = self._row(conn, task_id)
            raw = self._validate_identity(row, task_id, actor, expected_version, allow_cancelled=True)
            state, budget = self._decode(raw, task_id, actor)
            pending = budget.to_dict()['reservations'].get(state.inflight) if state.inflight else None
            return {'version': row['version'], 'cancelled': True,
                    'inflight': state.inflight or None,
                    'kind': pending['kind'] if pending else None}

    def reserve(self, task_id: str, actor: str, *, expected_version: int,
                reservation_id: str, kind: str, amounts: dict, now_ms: int,
                lane: str = 'work') -> Checkpoint:
        rejection = None
        result = None
        with self.store._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = self._row(conn, task_id)
            raw = self._validate_identity(row, task_id, actor, expected_version)
            state, budget = self._decode(raw, task_id, actor)
            if 'single_loop_model_claim' in raw['metadata']:
                raise CheckpointConflict('claim_model_binding_unresolved')
            if state.inflight:
                raise CheckpointConflict('single_loop_inflight_requires_reconciliation')
            if budget.usage()['unknown_usage']:
                raise CheckpointConflict('single_loop_model_usage_unresolved')
            if any(
                row['kind'] == 'model' and row['status'] == 'settled'
                and any(row['actual'][key] > row['reserved'][key] for key in ('input_tokens', 'output_tokens'))
                for row in budget.to_dict()['reservations'].values()
            ):
                raise CheckpointConflict('single_loop_model_reservation_overrun')
            original_revision = budget.revision
            try:
                budget.reserve(reservation_id, kind=kind, amounts=amounts, now_ms=now_ms, lane=lane)
            except BudgetExhausted as exc:
                rejection = exc
                if budget.revision != original_revision:
                    result = self._write(conn, row, raw, state, budget)
            else:
                state.inflight = reservation_id
                result = self._write(conn, row, raw, state, budget)
        # Rejected time observations must commit before returning the rejection.
        if rejection is not None:
            raise rejection
        assert result is not None
        return result

    def settle(self, task_id: str, actor: str, *, expected_version: int,
               reservation_id: str, outcome: str, input_tokens: int | None = None,
               output_tokens: int | None = None, evidence_ref: str | None = None) -> Checkpoint:
        with self.store._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = self._row(conn, task_id)
            raw = self._validate_identity(row, task_id, actor, expected_version)
            state, budget = self._decode(raw, task_id, actor)
            if 'single_loop_model_claim' in raw['metadata']:
                raise CheckpointConflict('claim_model_fenced_settlement_required')
            if not state.inflight or state.inflight != reservation_id:
                raise CheckpointConflict('single_loop_inflight_mismatch')
            pending = budget.to_dict()['reservations'][reservation_id]
            # This persistence adapter cannot independently prove that an
            # unsuccessful or cancelled dispatch had no side effects. Leave it
            # in flight until an explicit reconciliation contract exists.
            if outcome == 'cancelled' or (pending['kind'] == 'tool' and outcome != 'success'):
                raise CheckpointConflict('single_loop_inflight_requires_reconciliation')
            if pending['kind'] == 'tool':
                # A caller-supplied ID is not evidence of existence, visibility,
                # freshness or a result tied to this invocation. Never clear a
                # tool marker until those properties can be verified in the same
                # durable transition.
                raise CheckpointConflict('single_loop_verified_evidence_integration_required')
            if evidence_ref is not None:
                raise ValueError('single_loop_unexpected_evidence_reference')
            budget.settle(reservation_id, outcome=outcome, input_tokens=input_tokens, output_tokens=output_tokens)
            state.inflight = ''
            return self._write(conn, row, raw, state, budget)
