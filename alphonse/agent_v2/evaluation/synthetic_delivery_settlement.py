"""Disposable, single-database delivery-settlement experiment.

These fixture tables stand in for a trusted provider and an authority store.
They are not authenticated provider receipts, a production access policy, or a
recovery API. Never install them in a household database or connect this module
to a runtime. In particular, the test writer can forge any fixture evidence.
"""
from __future__ import annotations

from dataclasses import asdict
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
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimReceipt, _time, fenced_claim_transaction
from alphonse.agent_v2.core.io.experimental_staged_outbox import _validate_dispatch_schema, _validate_guards


def settle_fixture_delivery(store: ExperimentalCheckpointStore, claim: ClaimReceipt, *,
                            actor: str, task_id: str, expected_version: int,
                            clock: Callable[[], datetime]):
    """Settle only a matching fixture-provider delivery and current fixture grant.

    The trusted test setup independently seeds ``fixture_delivery_effects`` and
    ``fixture_delivery_grants``. This function accepts no receipt or success
    assertion from its caller. It performs no dispatch, retry, publication, or
    task completion, and does not establish real provider or user authority.
    """
    if (not isinstance(store, ExperimentalCheckpointStore) or not isinstance(claim, ClaimReceipt)
            or not isinstance(actor, str) or not actor.strip() or not isinstance(task_id, str)
            or not task_id.strip() or type(expected_version) is not int or expected_version < 1
            or not callable(clock) or claim.user_id != actor):
        raise ClaimFenceUnavailable('fixture_settlement_unavailable')
    try:
        with store.store._connect() as conn:
            main = next((row for row in conn.execute('PRAGMA database_list') if row[1] == 'main'), None)
            if main is None or not main[2]:
                raise ClaimFenceUnavailable('fixture_settlement_unavailable')
            with fenced_claim_transaction(conn, claim, clock=clock):
                return _settle_fixture_locked(conn, store, claim, actor=actor, task_id=task_id,
                                              expected_version=expected_version)
    except sqlite3.Error:
        raise ClaimFenceUnavailable('fixture_settlement_unavailable') from None


def _settle_fixture_locked(conn, store, claim, *, actor, task_id, expected_version):
    if conn.execute('SELECT 1 FROM main.v2_users WHERE user_id=? AND is_active=1', (actor,)).fetchone() is None:
        raise ClaimFenceUnavailable('fixture_settlement_unavailable')
    row = store._row(conn, task_id)
    raw = store._validate_identity(row, task_id, actor, expected_version)
    # A fixture effect and grant cannot prove the current authority
    # of a managed action, even if their values happen to match.
    if MANAGED_PREFLIGHT_KEY in raw['metadata']:
        raise ClaimFenceUnavailable('fixture_settlement_managed_unsupported')
    state, budget = store._decode(raw, task_id, actor)
    record = raw['metadata'].get(ACTION_KEY)
    keys = {'action_id', 'integration_id', 'provider_key', 'channel_target', 'recipient_user_id',
            'message_digest', 'idempotency_key', 'grant_id', 'grant_revision', 'connection_revision'}
    if (raw.get('message_id') != claim.message_id or not isinstance(record, dict)
            or set(record) != keys | {'schema_version', 'operation', 'task_id', 'actor', 'authority_mode', 'claim_digest'}
            or type(record['schema_version']) is not int or record['schema_version'] != ACTION_SCHEMA_VERSION
            or record['operation'] != 'delivery' or record['task_id'] != task_id or record['actor'] != actor
            or record['authority_mode'] != 'synthetic' or not delivery_claim_matches(record, claim)):
        raise ClaimFenceUnavailable('fixture_settlement_unavailable')
    binding = DeliveryBinding(**{key: record[key] for key in keys})
    pending = budget.to_dict()['reservations'].get(state.inflight)
    if state.inflight != binding.action_id or not pending or pending['kind'] != 'tool' or pending['status'] != 'reserved':
        raise ClaimFenceUnavailable('fixture_settlement_unavailable')
    # A fixture effect alone is not evidence that this acquisition
    # recorded a dispatch attempt. The immutable marker is created
    # in a separate earlier transaction, before fixture dispatch.
    _validate_dispatch_schema(conn)
    staged_id = str(uuid5(NAMESPACE_URL, 'alphonse-experimental-stage:' +
                          json.dumps([task_id, binding.idempotency_key], ensure_ascii=True,
                                     separators=(',', ':'))))
    marker = conn.execute('''SELECT task_id,actor,action_id,claim_digest,status
        FROM main.experimental_staged_dispatch_attempts WHERE outbox_message_id=?''',
        (staged_id,)).fetchone()
    if marker is None or tuple(marker) != (task_id, actor, binding.action_id,
                                           record['claim_digest'], 'started'):
        raise ClaimFenceUnavailable('fixture_settlement_unavailable')
    # These two rows are fixture-owned, not model or caller input.
    grants = conn.execute('''SELECT actor,recipient,integration,provider,channel,grant_id,revision,connection_revision,active
        FROM main.fixture_delivery_grants WHERE actor=? AND recipient=? AND integration=? AND provider=? AND channel=?''',
        (actor, binding.recipient_user_id, binding.integration_id, binding.provider_key, binding.channel_target)).fetchall()
    if len(grants) != 1 or tuple(grants[0]) != (actor, binding.recipient_user_id, binding.integration_id,
                                                binding.provider_key, binding.channel_target, binding.grant_id,
                                                binding.grant_revision, binding.connection_revision, 1):
        raise ClaimFenceUnavailable('fixture_settlement_unavailable')
    effects = conn.execute('''SELECT task_id,actor,action_id,integration,provider,channel,recipient,
               message_digest,idempotency_key,grant_id,grant_revision,connection_revision,status
        FROM main.fixture_delivery_effects WHERE task_id=? AND action_id=?''', (task_id, binding.action_id)).fetchall()
    expected = (task_id, actor, binding.action_id, binding.integration_id, binding.provider_key,
                binding.channel_target, binding.recipient_user_id, binding.message_digest,
                binding.idempotency_key, binding.grant_id, binding.grant_revision,
                binding.connection_revision, 'delivered')
    if len(effects) != 1 or tuple(effects[0]) != expected:
        raise ClaimFenceUnavailable('fixture_settlement_unavailable')
    budget.settle(binding.action_id, outcome='success')
    state.inflight = ''
    raw['metadata']['fixture_delivery_settlement'] = {'schema_version': 1, 'binding': asdict(binding),
                                                      'status': 'fixture_delivered'}
    return store._write(conn, row, raw, state, budget)


def _validate_recovery_staging(conn, raw, *, task_id, actor, expected_version):
    """Require the exact inert staged response, not merely an attempt marker."""
    _validate_guards(conn)
    record = raw['metadata'].get(ACTION_KEY)
    if not isinstance(record, dict):
        raise ClaimFenceUnavailable('fixture_recovery_unavailable')
    keys = DeliveryBinding.__dataclass_fields__
    binding = DeliveryBinding(**{key: record[key] for key in keys})
    staged_id = str(uuid5(NAMESPACE_URL, 'alphonse-experimental-stage:' +
                          json.dumps([task_id, binding.idempotency_key], ensure_ascii=True,
                                     separators=(',', ':'))))
    staged = conn.execute('''SELECT integration_id,provider_key,channel_target,message,kind,
        audience_user_id,correlation_id,status,provider_message_id,reply_to_provider_message_id,
        task_id,question_id,project_id,metadata_json,claimed_at,delivered_at,created_at,updated_at,
        attempt_count,lease_owner,lease_expires_at,next_attempt_at,last_error,max_attempts
        FROM main.v2_outbox WHERE outbox_message_id=?''', (staged_id,)).fetchone()
    metadata = json.dumps({'experimental_staged': True, 'checkpoint_version': expected_version,
                           'action_id': binding.action_id,
                           'idempotency_key_sha256': hashlib.sha256(binding.idempotency_key.encode('utf-8')).hexdigest()},
                          sort_keys=True)
    if staged is None or not isinstance(staged[3], str):
        raise ClaimFenceUnavailable('fixture_recovery_unavailable')
    row = tuple(staged)
    expected = (binding.integration_id, binding.provider_key, binding.channel_target, row[3], 'response', actor,
                '', 'staged', '', '', task_id, '', '', metadata, '', '')
    if (hashlib.sha256(row[3].encode('utf-8')).hexdigest() != binding.message_digest
            or binding.recipient_user_id != actor or row[:16] != expected
            or not isinstance(row[16], str) or not row[16] or row[17] != row[16]
            or row[18:] != (0, '', '', '', '', 5)):
        raise ClaimFenceUnavailable('fixture_recovery_unavailable')


def recover_expired_fixture_effect(store: ExperimentalCheckpointStore, *, actor: str, task_id: str,
                                   expected_version: int, clock: Callable[[], datetime]):
    """Reconcile one expired acquisition using disposable, forgeable fixture rows.

    A fresh process can call this without a serialized receipt. It reconstructs
    the original acquisition privately from the still-processing queue row, and
    the action and immutable attempt must bind to that exact generation. A
    separate active fixture recovery grant, current fixture delivery grant and
    exactly one matching fixture effect are required. No claim is acquired,
    extended, replaced, acknowledged or exposed; no dispatch, retry or task
    continuation occurs. Missing evidence stays pending.

    This does not authenticate ``actor`` or establish real provider evidence or
    authority. Fixture tables and trusted clocks are forgeable, old writers can
    bypass this protocol, and unobserved clock rollback is not detected. Never
    use this function or its schema for household data.
    """
    if (not isinstance(store, ExperimentalCheckpointStore) or not isinstance(actor, str) or not actor.strip()
            or not isinstance(task_id, str) or not task_id.strip() or type(expected_version) is not int
            or expected_version < 1 or not callable(clock)):
        raise ClaimFenceUnavailable('fixture_recovery_unavailable')
    try:
        with store.store._connect() as conn:
            main = next((row[2] for row in conn.execute('PRAGMA database_list') if row[1] == 'main'), '')
            if not main or not os.path.samefile(main, store.store.db_path) or conn.in_transaction:
                raise ClaimFenceUnavailable('fixture_recovery_unavailable')
            conn.execute('BEGIN IMMEDIATE')
            try:
                def sample_clock():
                    try:
                        return _time(clock())
                    except Exception:
                        raise ClaimFenceUnavailable('fixture_recovery_unavailable') from None
                start = sample_clock()
                row = store._row(conn, task_id)
                before_task = store._validate_identity(row, task_id, actor, expected_version)
                before_checkpoint = dict(conn.execute('SELECT * FROM main.v2_task_checkpoints WHERE task_id=?',
                                                      (task_id,)).fetchone())
                _validate_recovery_staging(conn, before_task, task_id=task_id, actor=actor,
                                           expected_version=expected_version)
                message_id = before_task.get('message_id')
                if not isinstance(message_id, str) or not message_id.strip():
                    raise ClaimFenceUnavailable('fixture_recovery_unavailable')
                queue_row = conn.execute('SELECT * FROM main.v2_inbound_messages WHERE message_id=?',
                                         (message_id,)).fetchone()
                if queue_row is None or queue_row['status'] != 'processing' or queue_row['user_id'] != actor:
                    raise ClaimFenceUnavailable('fixture_recovery_unavailable')
                expires = _time(datetime.fromisoformat(queue_row['lease_expires_at']))
                if expires > start:
                    raise ClaimFenceUnavailable('fixture_recovery_unavailable')
                claim = ClaimReceipt(message_id, actor, queue_row['lease_owner'], queue_row['claim_generation'])
                # Snapshot independently stored evidence and authority. A
                # trigger fired by checkpoint update must not change them.
                def evidence_snapshot():
                    return tuple(tuple(tuple(r) for r in conn.execute(query)) for query in (
                        'SELECT * FROM main.v2_inbound_messages ORDER BY rowid',
                        'SELECT * FROM main.fixture_delivery_grants ORDER BY rowid',
                        'SELECT * FROM main.fixture_delivery_effects ORDER BY rowid',
                        'SELECT * FROM main.fixture_recovery_grants ORDER BY rowid',
                        'SELECT * FROM main.experimental_staged_dispatch_attempts ORDER BY rowid',
                        'SELECT * FROM main.v2_outbox ORDER BY rowid',
                    ))
                grants = conn.execute('SELECT actor,task_id,active FROM main.fixture_recovery_grants '
                                      'WHERE actor=? AND task_id=?', (actor, task_id)).fetchall()
                if len(grants) != 1 or tuple(grants[0]) != (actor, task_id, 1):
                    raise ClaimFenceUnavailable('fixture_recovery_unavailable')
                before_evidence = evidence_snapshot()
                result = _settle_fixture_locked(conn, store, claim, actor=actor, task_id=task_id,
                                                expected_version=expected_version)
                after_task = store._validate_identity(store._row(conn, task_id), task_id, actor, result.version)
                _validate_recovery_staging(conn, after_task, task_id=task_id, actor=actor,
                                           expected_version=expected_version)
                state, budget = store._decode(after_task, task_id, actor)
                if state.snapshot() != result.state.snapshot() or budget.to_dict() != result.budget.to_dict():
                    raise ClaimFenceUnavailable('fixture_recovery_unavailable')
                after_checkpoint = dict(conn.execute('SELECT * FROM main.v2_task_checkpoints WHERE task_id=?',
                                                     (task_id,)).fetchone())
                for column in ('version', 'task_state_json', 'updated_at'):
                    before_checkpoint.pop(column, None)
                    after_checkpoint.pop(column, None)
                action = before_task['metadata'][ACTION_KEY]
                binding_keys = ('action_id', 'integration_id', 'provider_key', 'channel_target', 'recipient_user_id',
                                'message_digest', 'idempotency_key', 'grant_id', 'grant_revision', 'connection_revision')
                expected_settlement = {'schema_version': 1, 'binding': {key: action[key] for key in binding_keys},
                                       'status': 'fixture_delivered'}
                # Apart from the single checkpoint envelope and settlement
                # record, no task content may change during this transition.
                expected_task = json.loads(json.dumps(before_task))
                expected_task['metadata']['single_loop_checkpoint'] = after_task['metadata']['single_loop_checkpoint']
                expected_task['metadata']['fixture_delivery_settlement'] = expected_settlement
                if (after_task != expected_task or before_checkpoint != after_checkpoint
                        or evidence_snapshot() != before_evidence or conn.execute(
                        'SELECT 1 FROM main.v2_users WHERE user_id=? AND is_active=1', (actor,)).fetchone() is None):
                    raise ClaimFenceUnavailable('fixture_recovery_unavailable')
                end = sample_clock()
                if end < start:
                    raise ClaimFenceUnavailable('fixture_recovery_unavailable')
                conn.commit()
                return result
            except BaseException:
                if conn.in_transaction:
                    conn.rollback()
                raise
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError, StopIteration):
        raise ClaimFenceUnavailable('fixture_recovery_unavailable') from None
