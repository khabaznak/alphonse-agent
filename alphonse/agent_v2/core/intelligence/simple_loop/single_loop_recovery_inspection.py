"""Read-only recovery status for experimental single-loop checkpoints.

This is not wired into a runtime, IPC, or a recovery action. AccountSession is
trusted internal context, not authentication of an arbitrary caller. No status
here authorizes settlement, refund, retry, release, publication, or dispatch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import os
import re
import sqlite3
from typing import Callable

from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.core.messages.claim_fence import (
    ClaimFenceUnavailable, ClaimReceipt, ClaimSuperseded, _time, fenced_claim_transaction,
)
from .single_loop_checkpoint import CheckpointConflict, ExperimentalCheckpointStore
from .single_loop_delivery_reservation import ACTION_KEY, ACTION_SCHEMA_VERSION, DeliveryBinding, delivery_claim_matches


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
            # This inspector cannot validate a claim-bound model marker and
            # must not present an orphaned or unsupported one as clear.
            if 'single_loop_model_claim' in raw['metadata']:
                return unavailable
            from .single_loop_managed_delivery import MANAGED_PREFLIGHT_KEY, managed_record_matches
            # A standalone delivery action is not evidence that its effect was
            # reconciled. Malformed or mismatched action state must not be
            # projected as "no_pending" (nor as a model-only reservation).
            # This read-only inspector does not accept settlement assertions.
            if ACTION_KEY in raw['metadata']:
                record = raw['metadata'][ACTION_KEY]
                keys = set(DeliveryBinding.__dataclass_fields__)
                if (not isinstance(record, dict) or set(record) != keys | {'schema_version', 'operation', 'task_id', 'actor', 'authority_mode', 'claim_digest'}
                        or type(record.get('schema_version')) is not int
                        or record['schema_version'] != ACTION_SCHEMA_VERSION
                        or not isinstance(record['claim_digest'], str)
                        or re.fullmatch(r'[0-9a-f]{64}', record['claim_digest']) is None
                        or record['operation'] != 'delivery' or record['task_id'] != task_id
                        or record['actor'] != actor):
                    return unavailable
                binding = DeliveryBinding(**{key: record[key] for key in keys})
                reservation = budget.to_dict()['reservations'].get(state.inflight)
                if (state.inflight != binding.action_id or not isinstance(reservation, dict)
                        or reservation.get('kind') != 'tool' or reservation.get('status') != 'reserved'):
                    return unavailable
                if (record['authority_mode'] not in {'synthetic', 'managed'}
                        or (record['authority_mode'] == 'managed') != (MANAGED_PREFLIGHT_KEY in raw['metadata'])
                        or (record['authority_mode'] == 'managed' and not managed_record_matches(
                            raw, task_id=task_id, actor=actor, binding=binding))):
                    return unavailable
            elif MANAGED_PREFLIGHT_KEY in raw['metadata']:
                return unavailable
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
                if 'single_loop_model_claim' in raw['metadata']:
                    return unavailable
                from .single_loop_managed_delivery import MANAGED_PREFLIGHT_KEY, managed_record_matches
                record = raw['metadata'].get(ACTION_KEY)
                if not isinstance(record, dict):
                    return unavailable
                mode = record.get('authority_mode')
                if (mode not in {'synthetic', 'managed'}
                        or (mode == 'managed') != (MANAGED_PREFLIGHT_KEY in raw['metadata'])
                        or (mode == 'managed' and not managed_record_matches(
                            raw, task_id=task_id, actor=actor, binding=expected))):
                    return unavailable
                if not delivery_claim_matches(record, claim):
                    return unavailable
                expected_record = {'schema_version': ACTION_SCHEMA_VERSION, 'operation': 'delivery',
                                   'task_id': task_id, 'actor': actor, 'authority_mode': mode, **asdict(expected)}
                expected_record['claim_digest'] = record['claim_digest']
                pending = budget.to_dict()['reservations'].get(state.inflight)
                if (raw.get('message_id') != claim.message_id
                        or not isinstance(record, dict) or type(record.get('schema_version')) is not int
                        or type(record.get('grant_revision')) is not int
                        or type(record.get('connection_revision')) is not int
                        or record != expected_record
                        or state.inflight != expected.action_id
                        or not isinstance(pending, dict) or pending.get('kind') != 'tool'
                        or pending.get('status') != 'reserved'):
                    return unavailable
                return RecoveryInspection('binding_matches', 'running', 'tool')
    except (sqlite3.Error, ClaimFenceUnavailable, ClaimSuperseded, CheckpointConflict,
            ValueError, KeyError, TypeError):
        return unavailable


def inspect_managed_delivery_authority(store: ExperimentalCheckpointStore, session: AccountSession,
                                       claim: ClaimReceipt, *, bindings, task_id: str,
                                       expected_version: int, clock: Callable[[], datetime]) -> RecoveryInspection:
    """Recheck the recorded managed authority, without permitting any effect.

    The caller must authenticate the session. This unwired inspection uses the
    real resolver and fences its database snapshot against the current claim and
    checkpoint before and after resolution. Skill files and the credential vault
    are external to the SQLite lock and may change immediately after the check;
    a positive status is not a token, provider evidence, or permission to settle.
    The clock must be trusted; unobserved forward-then-backward changes cannot
    be detected. Observed regression across either transaction fails closed.
    """
    from alphonse.agent_v2.skill_bindings import SkillBindings
    from .single_loop_managed_delivery import MANAGED_PREFLIGHT_KEY, _capture, managed_record_matches

    if (not isinstance(store, ExperimentalCheckpointStore) or not isinstance(session, AccountSession)
            or not isinstance(bindings, SkillBindings) or not isinstance(claim, ClaimReceipt)
            or not isinstance(session.actor_user_id, str) or not session.actor_user_id.strip()
            or not isinstance(task_id, str) or not task_id.strip()
            or type(expected_version) is not int or expected_version < 1 or not callable(clock)):
        raise ValueError('recovery_inspection_arguments_invalid')
    unavailable = RecoveryInspection('unavailable')
    actor = session.actor_user_id
    if claim.user_id != actor:
        return unavailable
    previous = None
    def checked_clock():
        nonlocal previous
        try:
            now = _time(clock())
        except Exception:
            raise ClaimFenceUnavailable('managed_recovery_clock_unavailable') from None
        if previous is not None and now < previous:
            raise ClaimFenceUnavailable('managed_recovery_clock_regressed')
        previous = now
        return now
    try:
        if not (os.path.samefile(store.store.db_path, bindings.integrations.db_path)
                and os.path.samefile(store.store.db_path, bindings.users.db_path)):
            return unavailable

        def snapshot(conn):
            row = store._row(conn, task_id)
            raw = store._validate_identity(row, task_id, actor, expected_version)
            state, budget = store._decode(raw, task_id, actor)
            if 'single_loop_model_claim' in raw['metadata']:
                raise ClaimFenceUnavailable('managed_recovery_unavailable')
            record = raw['metadata'].get(ACTION_KEY)
            keys = set(DeliveryBinding.__dataclass_fields__)
            if (raw.get('message_id') != claim.message_id or not isinstance(record, dict)
                    or set(record) != keys | {'schema_version', 'operation', 'task_id', 'actor', 'authority_mode', 'claim_digest'}
                    or type(record.get('schema_version')) is not int or record['schema_version'] != ACTION_SCHEMA_VERSION
                    or record.get('operation') != 'delivery' or record.get('task_id') != task_id
                    or record.get('actor') != actor or record.get('authority_mode') != 'managed'
                    or not delivery_claim_matches(record, claim)):
                raise ClaimFenceUnavailable('managed_recovery_unavailable')
            binding = DeliveryBinding(**{key: record[key] for key in keys})
            pending = budget.to_dict()['reservations'].get(state.inflight)
            if (state.inflight != binding.action_id or not isinstance(pending, dict)
                    or pending.get('kind') != 'tool' or pending.get('status') != 'reserved'
                    or not managed_record_matches(raw, task_id=task_id, actor=actor, binding=binding)):
                raise ClaimFenceUnavailable('managed_recovery_unavailable')
            managed = raw['metadata'][MANAGED_PREFLIGHT_KEY]
            authority, visible = _capture(conn, actor=actor, skill_id=managed['skill_id'], role=managed['role'],
                                          binding_id=managed['binding_id'], action=binding)
            if any(visible[key] != managed[key] for key in ('binding_revision', 'required_scopes', 'required_executables')):
                raise ClaimFenceUnavailable('managed_recovery_unavailable')
            return raw, binding, managed, authority

        with store.store._connect() as conn:
            with fenced_claim_transaction(conn, claim, clock=checked_clock):
                original, binding, managed, before = snapshot(conn)
        handle = bindings.resolve_account(session, skill_id=managed['skill_id'], name=managed['role'],
            operation=managed['operation'], resources=managed['resources'],
            recipient_user_ids=managed['recipient_user_ids'], project_id=managed['project_id'],
            expected_binding_id=managed['binding_id'], expected_connection_id=binding.integration_id,
            expected_grant_id=binding.grant_id, expected_grant_revision=binding.grant_revision)
        if (handle.binding_id != managed['binding_id'] or handle.connection_id != binding.integration_id
                or handle.grant_id != binding.grant_id or handle.provider_key != binding.provider_key
                or handle.connection_revision != binding.connection_revision
                or handle.grant_revision != binding.grant_revision or handle.operation != managed['operation']
                or list(handle.resources) != managed['resources']
                or list(handle.recipient_user_ids) != managed['recipient_user_ids']):
            return unavailable
        with store.store._connect() as conn:
            with fenced_claim_transaction(conn, claim, clock=checked_clock):
                current, _, _, after = snapshot(conn)
                if current != original or after != before:
                    return unavailable
        return RecoveryInspection('authority_checked', 'running', 'tool')
    except (OSError, sqlite3.Error, ClaimFenceUnavailable, ClaimSuperseded, CheckpointConflict,
            PermissionError, ValueError, KeyError, TypeError, AttributeError):
        return unavailable
