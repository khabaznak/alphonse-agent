"""Unwired synthetic delivery reservation; no authorization or dispatch.

This prototype atomically binds one candidate tool reservation to a concrete
delivery and a live direct-owner claim. It cannot settle, retry or send. An
authenticated host must independently authorize the actor, recipient and grant
at dispatch and reconciliation; old writers and external I/O are not fenced.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
from typing import Callable

from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimReceipt, fenced_claim_transaction
from .single_loop_checkpoint import Checkpoint, CheckpointConflict, ExperimentalCheckpointStore
from .task_budget import BudgetExhausted


ACTION_KEY = 'single_loop_delivery_action'


@dataclass(frozen=True)
class DeliveryBinding:
    """Immutable exact operation inputs; never interpret a digest as authority."""
    action_id: str
    integration_id: str
    provider_key: str
    channel_target: str
    recipient_user_id: str
    message_digest: str
    idempotency_key: str
    grant_revision: str

    def __post_init__(self) -> None:
        import re
        values = asdict(self)
        if any(not isinstance(v, str) or not v.strip() or len(v) > 512 for v in values.values()):
            raise ValueError('delivery_binding_invalid')
        if re.fullmatch(r'[0-9a-f]{64}', self.message_digest) is None:
            raise ValueError('delivery_binding_digest_invalid')


def reserve_delivery(store: ExperimentalCheckpointStore, claim: ClaimReceipt, *, task_id: str,
                     actor: str, expected_version: int, binding: DeliveryBinding,
                     message: str, now_ms: int, clock: Callable[[], datetime]) -> Checkpoint:
    """Reserve a delivery atomically; caller must not dispatch using this alone.

    Only a direct-owner queue message and file-backed shared checkpoint are
    supported. Existing legacy reservations and action bindings are not adopted.
    The idempotency key is not globally enforced. The caller-supplied grant
    revision is not authorization. The message is not retained here, so a host
    must also retain durable authorized inputs before any dispatch. This prototype
    never clears or reconciles the resulting marker and must not be used to dispatch.
    """
    if (not isinstance(store, ExperimentalCheckpointStore) or not isinstance(claim, ClaimReceipt)
            or not isinstance(binding, DeliveryBinding) or not callable(clock)
            or not isinstance(message, str) or not message or len(message.encode('utf-8')) > 12000):
        raise ValueError('delivery_reservation_arguments_invalid')
    # Hash exactly the UTF-8 bytes supplied for the intended message. Do not
    # normalize whitespace or Unicode: an independent verifier must compare the
    # precise operation input, not an agent-supplied assertion about its digest.
    if hashlib.sha256(message.encode('utf-8')).hexdigest() != binding.message_digest:
        raise ValueError('delivery_reservation_message_mismatch')
    if claim.user_id != actor:
        raise ClaimFenceUnavailable('delivery_reservation_unavailable')
    rejected = None
    result = None
    with store.store._connect() as conn:
        main = next((row for row in conn.execute('PRAGMA database_list') if row[1] == 'main'), None)
        if main is None or not main[2]:
            raise ClaimFenceUnavailable('delivery_reservation_shared_file_required')
        with fenced_claim_transaction(conn, claim, clock=clock):
            if conn.execute('SELECT 1 FROM main.v2_users WHERE user_id=? AND is_active=1', (actor,)).fetchone() is None:
                raise ClaimFenceUnavailable('delivery_reservation_unavailable')
            row = store._row(conn, task_id)
            raw = store._validate_identity(row, task_id, actor, expected_version)
            if raw.get('message_id') != claim.message_id or raw.get('status') != 'running':
                raise ClaimFenceUnavailable('delivery_reservation_unavailable')
            state, budget = store._decode(raw, task_id, actor)
            if state.inflight or ACTION_KEY in raw['metadata']:
                raise CheckpointConflict('delivery_action_already_bound_or_inflight')
            if budget.usage()['unknown_usage']:
                raise CheckpointConflict('single_loop_model_usage_unresolved')
            if any(r['kind'] == 'model' and r['status'] == 'settled'
                   and any(r['actual'][key] > r['reserved'][key] for key in ('input_tokens', 'output_tokens'))
                   for r in budget.to_dict()['reservations'].values()):
                raise CheckpointConflict('single_loop_model_reservation_overrun')
            previous_revision = budget.revision
            try:
                budget.reserve(binding.action_id, kind='tool', now_ms=now_ms,
                               amounts={'actions': 1, 'model_calls': 0, 'input_tokens': 0, 'output_tokens': 0})
            except BudgetExhausted as exc:
                rejected = exc
                if budget.revision != previous_revision:
                    result = store._write(conn, row, raw, state, budget)
            else:
                state.inflight = binding.action_id
                raw['metadata'][ACTION_KEY] = {'schema_version': 1, 'operation': 'delivery',
                                               'task_id': task_id, 'actor': actor, **asdict(binding)}
                result = store._write(conn, row, raw, state, budget)
    if rejected is not None:
        raise rejected
    assert result is not None
    return result
