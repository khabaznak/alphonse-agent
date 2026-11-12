"""Read-only recovery status for experimental single-loop checkpoints.

This is not wired into a runtime, IPC, or a recovery action. AccountSession is
trusted internal context, not authentication of an arbitrary caller. No status
here authorizes settlement, refund, retry, release, publication, or dispatch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import sqlite3
from typing import Callable

from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.core.messages.claim_fence import (
    ClaimFenceUnavailable, ClaimReceipt, ClaimSuperseded, fenced_claim_transaction,
)
from .single_loop_checkpoint import CheckpointConflict, ExperimentalCheckpointStore
from .single_loop_delivery_reservation import ACTION_KEY, DeliveryBinding


@dataclass(frozen=True)
class RecoveryInspection:
    status: str
    checkpoint_status: str | None = None
    pending_kind: str | None = None


def inspect_recovery(store: ExperimentalCheckpointStore, session: AccountSession, *,
                     task_id: str, expected_version: int) -> RecoveryInspection:
    """Inspect a direct owner's running/cancelled checkpoint in one snapshot.

    Missing, stale, unauthorized, malformed, and inactive state is opaque.
    The caller must independently authenticate the session, recheck audience
    before presentation, and implement authorized reconciliation separately.
    """
    if (not isinstance(store, ExperimentalCheckpointStore) or not isinstance(session, AccountSession)
            or not isinstance(session.actor_user_id, str) or not session.actor_user_id.strip()
            or not isinstance(task_id, str) or not task_id.strip()
            or type(expected_version) is not int or expected_version < 1):
        raise ValueError('recovery_inspection_arguments_invalid')
    unavailable = RecoveryInspection('unavailable')
    actor = session.actor_user_id
    try:
        with store.store._connect() as conn:
            conn.execute('BEGIN')
            if conn.execute('SELECT 1 FROM main.v2_users WHERE user_id=? AND is_active=1', (actor,)).fetchone() is None:
                return unavailable
            row = store._row(conn, task_id)
            if row is None or row['status'] not in {'running', 'cancelled'}:
                return unavailable
            raw = store._validate_identity(row, task_id, actor, expected_version,
                                           allow_cancelled=row['status'] == 'cancelled')
            state, budget = store._decode(raw, task_id, actor)
            if not state.inflight:
                if budget.usage()['unknown_usage']:
                    return RecoveryInspection('usage_unresolved', row['status'], 'model')
                return RecoveryInspection('no_pending', row['status'])
            reservation = budget.to_dict()['reservations'][state.inflight]
            return RecoveryInspection('requires_reconciliation', row['status'], reservation['kind'])
    except (sqlite3.Error, CheckpointConflict, ValueError, KeyError, TypeError):
        return unavailable


def inspect_delivery_binding(store: ExperimentalCheckpointStore, session: AccountSession,
                             claim: ClaimReceipt, *, task_id: str, expected_version: int,
                             expected: DeliveryBinding, clock: Callable[[], datetime]) -> RecoveryInspection:
    """Check one pending delivery's structural binding under its current claim.

    This is read-only and unwired. The trusted caller must authenticate the
    session; a matching binding and grant revision prove neither current
    authority nor that a provider did or did not deliver. No result permits
    settlement, retry, publication, marker clearance or external I/O.
    """
    if (not isinstance(store, ExperimentalCheckpointStore) or not isinstance(session, AccountSession)
            or not isinstance(session.actor_user_id, str) or not session.actor_user_id.strip()
            or not isinstance(claim, ClaimReceipt) or not isinstance(expected, DeliveryBinding)
            or not isinstance(task_id, str) or not task_id.strip()
            or type(expected_version) is not int or expected_version < 1 or not callable(clock)):
        raise ValueError('recovery_inspection_arguments_invalid')
    unavailable = RecoveryInspection('unavailable')
    actor = session.actor_user_id
    if claim.user_id != actor:
        return unavailable
    try:
        with store.store._connect() as conn:
            main = next((row for row in conn.execute('PRAGMA database_list') if row[1] == 'main'), None)
            if main is None or not main[2]:
                return unavailable
            with fenced_claim_transaction(conn, claim, clock=clock):
                if conn.execute('SELECT 1 FROM main.v2_users WHERE user_id=? AND is_active=1', (actor,)).fetchone() is None:
                    return unavailable
                row = store._row(conn, task_id)
                raw = store._validate_identity(row, task_id, actor, expected_version)
                state, budget = store._decode(raw, task_id, actor)
                expected_record = {'schema_version': 1, 'operation': 'delivery',
                                   'task_id': task_id, 'actor': actor, **asdict(expected)}
                record = raw['metadata'].get(ACTION_KEY)
                pending = budget.to_dict()['reservations'].get(state.inflight)
                if (raw.get('message_id') != claim.message_id
                        or not isinstance(record, dict) or type(record.get('schema_version')) is not int
                        or record != expected_record
                        or state.inflight != expected.action_id
                        or not isinstance(pending, dict) or pending.get('kind') != 'tool'
                        or pending.get('status') != 'reserved'):
                    return unavailable
                return RecoveryInspection('binding_matches', 'running', 'tool')
    except (sqlite3.Error, ClaimFenceUnavailable, ClaimSuperseded, CheckpointConflict,
            ValueError, KeyError, TypeError):
        return unavailable
