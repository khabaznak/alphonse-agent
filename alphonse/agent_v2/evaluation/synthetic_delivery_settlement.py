"""Disposable, single-database delivery-settlement experiment.

These fixture tables stand in for a trusted provider and an authority store.
They are not authenticated provider receipts, a production access policy, or a
recovery API. Never install them in a household database or connect this module
to a runtime. In particular, the test writer can forge any fixture evidence.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
import sqlite3
from typing import Callable
from uuid import NAMESPACE_URL, uuid5

from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.v3.single_loop_delivery_reservation import ACTION_KEY, ACTION_SCHEMA_VERSION, DeliveryBinding, delivery_claim_matches
from alphonse.agent_v2.core.intelligence.v3.single_loop_managed_delivery import MANAGED_PREFLIGHT_KEY
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimReceipt, fenced_claim_transaction
from alphonse.agent_v2.core.io.experimental_staged_outbox import _validate_dispatch_schema


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
    except sqlite3.Error:
        raise ClaimFenceUnavailable('fixture_settlement_unavailable') from None
