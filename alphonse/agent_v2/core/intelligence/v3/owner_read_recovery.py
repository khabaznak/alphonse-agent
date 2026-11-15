"""Revalidate durable owner-read observations before candidate recovery.

This is a read-only component, not an enabled daemon recovery path. It never
replays an in-flight operation or settles an uncertain model call or delivery.
"""
from __future__ import annotations

import json
import hmac

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.tools.registry.native.memory import (
    READ_MEMORY_SOURCE_TOOL_ID, SEARCH_MEMORY_TOOL_ID, revalidate_conversation_observation,
    execute_search_memory, execute_read_memory_source, conversation_read_sequences,
)
from .owner_atomic_checkpoint import OwnerAtomicCheckpoint, observation_digest
from .owner_clock import OWNER_CLOCK_TOOL_ID, valid_clock_observation


class OwnerRecoveryUnavailable(Exception):
    """No durable owner state can safely be exposed for continuation."""


def restore_owner_read_observations(checkpoint: OwnerAtomicCheckpoint, context: CoreLoopContext, *, expected_version: int,
                                    allow_completed: bool = False):
    """Load a quiescent checkpoint and re-read each actor-bound source.

    The two claim-bound loads bracket source I/O, but cannot prevent source or
    authority changes immediately afterwards. The eventual host must recheck
    at every use and must fence subsequent writes. No uncertain call is retried.
    """
    if (type(checkpoint) is not OwnerAtomicCheckpoint or not isinstance(context, CoreLoopContext)
            or context.messages is not checkpoint.queue or context.claim_receipt is not checkpoint.receipt
            or not callable(context.cancellation_checker) or context.conversation_store is None
            or type(expected_version) is not int or expected_version < 1):
        raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
    try:
        if context.cancellation_checker() is not False:
            raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
        saved, bindings = checkpoint.load(expected_version=expected_version)
        state = saved.state
        if (type(allow_completed) is not bool
                or state.status not in ({'running', 'completed', 'waiting_user'} if allow_completed else {'running'})
                or state.inflight or saved.budget.usage()['unknown_usage']):
            raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
        # The checkpoint load refuses an uncommitted model decision. A known
        # call that exceeded its reservation must also stop on recovery.
        for reservation in saved.budget.to_dict()['reservations'].values():
            if (reservation['kind'] == 'model' and reservation['status'] == 'settled'
                    and any(reservation['actual'][key] > reservation['reserved'][key]
                            for key in ('input_tokens', 'output_tokens'))):
                raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
        # Recheck the live owner claim and exact checkpoint before every
        # source operation. A single check around the complete batch would
        # allow later reads after revocation during an earlier read. SQLite
        # cannot fence access to a separate conversation store between this
        # point-in-time check and the actual read.
        def canonical(value):
            return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)

        def recheck():
            if context.cancellation_checker() is not False:
                raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
            current, current_bindings = checkpoint.load(expected_version=expected_version)
            if (canonical(saved.state.snapshot()) != canonical(current.state.snapshot())
                    or canonical(saved.budget.to_dict()) != canonical(current.budget.to_dict())
                    or canonical(bindings) != canonical(current_bindings)):
                raise OwnerRecoveryUnavailable('owner_recovery_unavailable')

        execution = context.tool_execution_context(checkpoint.task)
        reconstructed = {}
        for ref in state.evidence_sources:
            item = bindings[ref]
            keys = {'actor', 'tool_id', 'arguments', 'observation_sha256'}
            if isinstance(item, dict) and item.get('tool_id') == OWNER_CLOCK_TOOL_ID:
                keys.add('clock_observation')
            if not isinstance(item, dict) or set(item) != keys:
                raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
            args = item['arguments']
            tool = item['tool_id']
            if (item['actor'] != state.actor or not isinstance(args, dict)
                    or not isinstance(item['observation_sha256'], str)):
                raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
            if tool == OWNER_CLOCK_TOOL_ID:
                recheck()
                observation = item['clock_observation']
                if not valid_clock_observation(state.actor, args, observation, context=execution):
                    raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
            elif tool == SEARCH_MEMORY_TOOL_ID:
                if args.get('source_types') != ['conversation'] or args.get('offset', 0) != 0:
                    raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
                recheck()
                observation = execute_search_memory(args, context=execution)
            elif tool == READ_MEMORY_SOURCE_TOOL_ID:
                if not isinstance(args.get('source_id'), str) or not args['source_id'].startswith('conversation:'):
                    raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
                recheck()
                observation = execute_read_memory_source(args, context=execution)
            else:
                raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
            # Revalidation itself re-reads the source, so authorize it
            # separately instead of continuing after a mid-read revocation.
            recheck()
            if (not hmac.compare_digest(observation_digest(observation), item['observation_sha256'])
                    or not (valid_clock_observation(state.actor, args, observation, context=execution)
                            if tool == OWNER_CLOCK_TOOL_ID else
                            revalidate_conversation_observation(state.actor, tool, args, observation,
                                                               context=execution, allow_excerpt_truncation=True,
                                                               allow_read_chunks=True, allow_search_overflow=True))):
                raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
            recheck()
            reconstructed[ref] = {'actor': state.actor, 'tool_id': tool, 'arguments': args, 'observation': observation}
        conversation_read_sequences(state.actor, {ref: (item['tool_id'], item['arguments'], item['observation'])
                                                   for ref, item in reconstructed.items()})
        recheck()
        return saved, reconstructed
    except Exception:
        raise OwnerRecoveryUnavailable('owner_recovery_unavailable') from None
