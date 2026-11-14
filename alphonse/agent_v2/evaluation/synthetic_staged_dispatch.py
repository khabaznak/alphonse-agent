"""Disposable same-database simulation; never connect to a household runtime.

Fixture grants and provider effects are forgeable test data. This experiment
does not authenticate actors, send a message, fence external I/O, or authorize
production settlement. The staged outbox record stays non-dispatchable.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
import sqlite3
from typing import Callable
from uuid import NAMESPACE_URL, uuid5

from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.v3.single_loop_delivery_reservation import ACTION_KEY, ACTION_SCHEMA_VERSION, DeliveryBinding, delivery_claim_matches
from alphonse.agent_v2.core.intelligence.v3.single_loop_managed_delivery import MANAGED_PREFLIGHT_KEY
from alphonse.agent_v2.core.io.checkpoint_claim_fence import experimental_checkpoint_claim_transaction
from alphonse.agent_v2.core.io.experimental_staged_outbox import _validate_guards
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimReceipt, _time


def dispatch_fixture_staged(store: SQLiteOutboundStore, checkpoints: ExperimentalCheckpointStore,
                            claim: ClaimReceipt, *, actor: str, task_id: str,
                            expected_version: int, staged_id: str,
                            clock: Callable[[], datetime]) -> None:
    """Commit an attempt before recording a synthetic effect in a second transaction.

    A crash between transactions leaves an uncertain attempt and cannot trigger
    a retry. This is fixture wiring, not an external I/O or authority guarantee.
    """
    if not callable(clock):
        raise ValueError('fixture_dispatch_arguments_invalid')
    # The claim fence validates ordering within each transaction. Share the
    # high-water mark across both transactions as well: a clock regression
    # between the committed attempt and effect must not revive an expired
    # claim. Unobserved regressions remain outside this fixture's guarantees.
    previous = None
    def checked_clock():
        nonlocal previous
        try:
            now = _time(clock())
        except Exception:
            raise ClaimFenceUnavailable('fixture_dispatch_clock_unavailable') from None
        if previous is not None and now < previous:
            raise ClaimFenceUnavailable('fixture_dispatch_clock_reversed')
        previous = now
        return now
    _fixture_dispatch_step(store, checkpoints, claim, actor=actor, task_id=task_id,
                           expected_version=expected_version, staged_id=staged_id, clock=checked_clock, begin=True)
    _fixture_dispatch_step(store, checkpoints, claim, actor=actor, task_id=task_id,
                           expected_version=expected_version, staged_id=staged_id, clock=checked_clock, begin=False)


def begin_fixture_dispatch(store: SQLiteOutboundStore, checkpoints: ExperimentalCheckpointStore,
                           claim: ClaimReceipt, *, actor: str, task_id: str,
                           expected_version: int, staged_id: str,
                           clock: Callable[[], datetime]) -> None:
    """Commit only the attempt, for a synthetic crash-before-effect test."""
    _fixture_dispatch_step(store, checkpoints, claim, actor=actor, task_id=task_id,
                           expected_version=expected_version, staged_id=staged_id, clock=clock, begin=True)


def _fixture_dispatch_step(store: SQLiteOutboundStore, checkpoints: ExperimentalCheckpointStore,
                           claim: ClaimReceipt, *, actor: str, task_id: str,
                           expected_version: int, staged_id: str,
                           clock: Callable[[], datetime], begin: bool) -> None:
    if (not isinstance(store, SQLiteOutboundStore) or store._memory_connection is not None
            or not isinstance(checkpoints, ExperimentalCheckpointStore)
            or not isinstance(claim, ClaimReceipt) or not callable(clock)
            or any(not isinstance(v, str) or not v.strip() for v in (actor, task_id, staged_id))
            or type(expected_version) is not int or expected_version < 1):
        raise ValueError('fixture_dispatch_arguments_invalid')
    try:
        if not os.path.samefile(store.db_path, checkpoints.store.db_path):
            raise ClaimFenceUnavailable('fixture_dispatch_unavailable')
        with store._connect() as conn:
            with experimental_checkpoint_claim_transaction(conn, claim, task_id=task_id, actor=actor,
                                                           expected_version=expected_version, clock=clock):
                _validate_guards(conn)
                if conn.execute('SELECT 1 FROM main.v2_users WHERE user_id=? AND is_active=1', (actor,)).fetchone() is None:
                    raise ClaimFenceUnavailable('fixture_dispatch_unavailable')
                row = checkpoints._row(conn, task_id)
                raw = checkpoints._validate_identity(row, task_id, actor, expected_version)
                # Fixture grants cannot authenticate or authorize an action that
                # carries a real managed preflight. Never let a synthetic grant
                # substitute for the managed account's current authority.
                if MANAGED_PREFLIGHT_KEY in raw['metadata']:
                    raise ClaimFenceUnavailable('fixture_dispatch_managed_unsupported')
                state, budget = checkpoints._decode(raw, task_id, actor)
                action = raw['metadata'].get(ACTION_KEY)
                keys = {'action_id', 'integration_id', 'provider_key', 'channel_target', 'recipient_user_id',
                        'message_digest', 'idempotency_key', 'grant_id', 'grant_revision', 'connection_revision'}
                if (not isinstance(action, dict) or set(action) != keys | {'schema_version', 'operation', 'task_id', 'actor', 'authority_mode', 'claim_digest'}
                        or type(action.get('schema_version')) is not int or action['schema_version'] != ACTION_SCHEMA_VERSION
                        or action['operation'] != 'delivery' or action['task_id'] != task_id or action['actor'] != actor
                        or action['authority_mode'] != 'synthetic' or not delivery_claim_matches(action, claim)):
                    raise ClaimFenceUnavailable('fixture_dispatch_unavailable')
                binding = DeliveryBinding(**{key: action[key] for key in keys})
                expected_id = str(uuid5(NAMESPACE_URL, 'alphonse-experimental-stage:' +
                                        json.dumps([task_id, binding.idempotency_key], ensure_ascii=True,
                                                   separators=(',', ':'))))
                if staged_id != expected_id:
                    raise ClaimFenceUnavailable('fixture_dispatch_unavailable')
                pending = budget.to_dict()['reservations'].get(state.inflight)
                if (state.inflight != binding.action_id or not isinstance(pending, dict)
                        or pending['status'] != 'reserved' or pending['kind'] != 'tool'):
                    raise ClaimFenceUnavailable('fixture_dispatch_unavailable')
                staged = conn.execute('''SELECT integration_id,provider_key,channel_target,message,kind,
                    audience_user_id,status,task_id,metadata_json,provider_message_id,claimed_at,delivered_at,
                    attempt_count,lease_owner,lease_expires_at,next_attempt_at,last_error
                    FROM main.v2_outbox WHERE outbox_message_id=?''', (staged_id,)).fetchone()
                expected_metadata = {'experimental_staged': True, 'checkpoint_version': expected_version,
                                     'action_id': binding.action_id,
                                     'idempotency_key_sha256': hashlib.sha256(binding.idempotency_key.encode('utf-8')).hexdigest()}
                try:
                    metadata = json.loads(staged[8]) if staged is not None else None
                except (TypeError, ValueError):
                    metadata = None
                if (staged is None or tuple(staged[:3]) != (binding.integration_id, binding.provider_key, binding.channel_target)
                        or not isinstance(staged[3], str)
                        or hashlib.sha256(staged[3].encode('utf-8')).hexdigest() != binding.message_digest
                        or tuple(staged[4:8]) != ('response', binding.recipient_user_id, 'staged', task_id)
                        or binding.recipient_user_id != actor or metadata != expected_metadata
                        or tuple(staged[9:]) != ('', '', '', 0, '', '', '', '')):
                    raise ClaimFenceUnavailable('fixture_dispatch_unavailable')
                grants = conn.execute('''SELECT actor,recipient,integration,provider,channel,grant_id,revision,connection_revision,active
                    FROM main.fixture_delivery_grants WHERE actor=? AND recipient=? AND integration=? AND provider=? AND channel=?''',
                    (actor, binding.recipient_user_id, binding.integration_id, binding.provider_key, binding.channel_target)).fetchall()
                expected_grant = (actor, binding.recipient_user_id, binding.integration_id, binding.provider_key,
                                  binding.channel_target, binding.grant_id, binding.grant_revision, binding.connection_revision, 1)
                if len(grants) != 1 or tuple(grants[0]) != expected_grant:
                    raise ClaimFenceUnavailable('fixture_dispatch_unavailable')
                if conn.execute('SELECT 1 FROM main.fixture_delivery_effects WHERE task_id=? AND action_id=?',
                                (task_id, binding.action_id)).fetchone() is not None:
                    raise ClaimFenceUnavailable('fixture_dispatch_already_observed')
                marker = conn.execute('''SELECT task_id,actor,action_id,claim_digest,status
                    FROM main.experimental_staged_dispatch_attempts WHERE outbox_message_id=?''', (staged_id,)).fetchone()
                expected_marker = (task_id, actor, binding.action_id, action['claim_digest'], 'started')
                if begin:
                    if marker is not None:
                        raise ClaimFenceUnavailable('fixture_dispatch_already_observed')
                    conn.execute('''INSERT INTO main.experimental_staged_dispatch_attempts
                        (outbox_message_id,task_id,actor,action_id,claim_digest,status) VALUES (?,?,?,?,?,'started')''',
                        (staged_id, task_id, actor, binding.action_id, action['claim_digest']))
                    return
                if marker is None or tuple(marker) != expected_marker:
                    raise ClaimFenceUnavailable('fixture_dispatch_unavailable')
                conn.execute('''INSERT INTO main.fixture_delivery_effects(task_id,actor,action_id,integration,
                    provider,channel,recipient,message_digest,idempotency_key,grant_id,grant_revision,connection_revision,status)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'delivered')''',
                    (task_id, actor, binding.action_id, binding.integration_id, binding.provider_key,
                     binding.channel_target, binding.recipient_user_id, binding.message_digest,
                     binding.idempotency_key, binding.grant_id, binding.grant_revision, binding.connection_revision))
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
        raise ClaimFenceUnavailable('fixture_dispatch_unavailable') from exc
