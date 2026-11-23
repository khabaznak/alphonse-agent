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
    execute_search_memory, execute_read_memory_source, conversation_read_sequences, conversation_search_sequences,
)
from .owner_atomic_checkpoint import OwnerAtomicCheckpoint, observation_digest
from .owner_clock import OWNER_CLOCK_TOOL_ID, valid_clock_observation
from .owner_reminder_create import OWNER_REMINDER_CREATE_TOOL_ID, valid_reminder_observation
from .owner_schedule_create import SCHEDULED_TASK_TOOL_ID, valid_schedule_observation
from .owner_schedule_cancel import valid_cancel_observation
from .owner_gmail import OWNER_GMAIL_TOOL_ID, valid_owner_gmail
from .owner_schedules import OWNER_SCHEDULES_TOOL_ID, execute_owner_schedules, valid_schedules_observation
from alphonse.agent_v2.core.tools.registry.native.skill_catalog import SKILL_CATALOG_TOOL_ID, execute_skill_catalog, validate_skill_catalog_observation
from .owner_named_skill import OWNER_NAMED_SKILL_TOOL_ID, execute_owner_named_skill, validate_owner_named_skill
from .owner_knowledge_snapshot import (OWNER_KNOWLEDGE_TOOL_ID, snapshot_owner_knowledge,
                                       owner_knowledge_projection)
from .dojo_bash_guard import valid_dojo_observation, valid_dojo_session
from .owner_web import WEB_TOOL_IDS, valid_web_observation


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
        # A reservation is an estimate; only actual aggregate work usage can
        # establish whether the task still fits while preserving final reserve.
        if not saved.budget.within_work_limits():
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
        # Once the atomic scheduled effect is committed, its original clock is
        # historical evidence, not a fresh instruction to schedule again.
        historical_schedule_clock = (allow_completed and state.status == 'completed'
            and any(isinstance(item, dict) and item.get('tool_id') == SCHEDULED_TASK_TOOL_ID
                    for item in bindings.values()))
        reconstructed = {}
        for ref in state.evidence_sources:
            item = bindings[ref]
            host_bash = (isinstance(item, dict) and item.get('tool_id') == 'native.bash'
                         and 'host_bash_observation' in item)
            keys = {'actor', 'tool_id', 'arguments', 'observation_sha256'}
            if isinstance(item, dict) and item.get('tool_id') == OWNER_CLOCK_TOOL_ID:
                keys.add('clock_observation')
            if isinstance(item, dict) and item.get('tool_id') == OWNER_GMAIL_TOOL_ID:
                keys.add('gmail_observation')
            if isinstance(item, dict) and item.get('tool_id') == OWNER_REMINDER_CREATE_TOOL_ID:
                keys.add('reminder_observation')
            if isinstance(item, dict) and item.get('tool_id') == SCHEDULED_TASK_TOOL_ID:
                keys.add('schedule_observation')
            if isinstance(item, dict) and item.get('tool_id') == OWNER_KNOWLEDGE_TOOL_ID:
                keys.add('knowledge_snapshot_digest')
            if isinstance(item, dict) and item.get('tool_id') in WEB_TOOL_IDS:
                keys.add('web_observation')
            if host_bash:
                keys.add('host_bash_observation')
            elif isinstance(item, dict) and item.get('tool_id') == 'native.bash':
                keys.update({'dojo_bash_observation', 'dojo_session_id'})
            if not isinstance(item, dict) or set(item) != keys:
                raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
            args = item['arguments']
            tool = item['tool_id']
            if (item['actor'] != state.actor or not isinstance(args, dict)
                    or not isinstance(item['observation_sha256'], str)):
                raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
            if tool in WEB_TOOL_IDS:
                recheck()
                observation = item['web_observation']
                if not valid_web_observation(state.actor, tool, args, observation,
                                             context=context, task=checkpoint.task):
                    raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
            elif tool == 'native.bash':
                recheck()
                if host_bash:
                    from .owner_host_bash import valid_host_binding
                    with checkpoint._transaction() as conn:
                        # Recovery may inspect a paused checkpoint, but cannot
                        # mutate it or dispatch while its pause marker exists.
                        _, raw, _, _ = checkpoint._load(conn, expected_version, allow_paused=True)
                        history = raw['metadata'].get('owner_host_bash_history', [])
                    if not valid_host_binding(checkpoint, context, item, history):
                        raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
                    observation = item['host_bash_observation']
                else:
                    if (not valid_dojo_session(context, item['dojo_session_id'])
                            or set(args) != {'command'} or not isinstance(args['command'], str)
                            or not args['command'].strip() or len(args['command'].encode()) > 16384):
                        raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
                    observation = item['dojo_bash_observation']
                    if not valid_dojo_observation(observation):
                        raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
            elif tool == OWNER_KNOWLEDGE_TOOL_ID:
                recheck()
                if args != {} or not isinstance(item['knowledge_snapshot_digest'], str):
                    raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
                snapshot = snapshot_owner_knowledge(checkpoint.queue, checkpoint.receipt, checkpoint.task,
                    cancellation_checker=context.cancellation_checker, clock=checkpoint.clock)
                observation = owner_knowledge_projection(snapshot)
                if (observation is None or not hmac.compare_digest(snapshot.digest, item['knowledge_snapshot_digest'])):
                    raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
            elif tool == OWNER_CLOCK_TOOL_ID:
                recheck()
                observation = item['clock_observation']
                if not valid_clock_observation(state.actor, args, observation, context=execution,
                                               allow_expired=historical_schedule_clock):
                    raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
            elif tool == OWNER_REMINDER_CREATE_TOOL_ID:
                recheck()
                observation = item['reminder_observation']
                if not valid_reminder_observation(state.actor, args, observation, context=execution):
                    raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
            elif tool == SCHEDULED_TASK_TOOL_ID:
                recheck()
                observation = item['schedule_observation']
                if not (valid_cancel_observation if args.get('operation') == 'cancel' else valid_schedule_observation)(
                        state.actor, args, observation, context=execution):
                    raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
            elif tool == OWNER_GMAIL_TOOL_ID:
                recheck()
                observation = item['gmail_observation']
                if not valid_owner_gmail(state.actor, args, observation, context=execution):
                    raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
            elif tool == OWNER_SCHEDULES_TOOL_ID:
                recheck()
                observation = execute_owner_schedules(args, context=execution)
            elif tool == SKILL_CATALOG_TOOL_ID:
                recheck()
                observation = execute_skill_catalog(args, context=execution)
            elif tool == OWNER_NAMED_SKILL_TOOL_ID:
                recheck()
                observation = execute_owner_named_skill(args, context=execution)
            elif tool == SEARCH_MEMORY_TOOL_ID:
                if args.get('source_types') != ['conversation'] or type(args.get('offset', 0)) is not int or args.get('offset', 0) < 0:
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
                    or not (valid_web_observation(state.actor, tool, args, observation,
                                                  context=context, task=checkpoint.task)
                            if tool in WEB_TOOL_IDS else
                            True if tool == 'native.bash' and host_bash else
                            valid_dojo_session(context, item['dojo_session_id']) and valid_dojo_observation(observation)
                            if tool == 'native.bash' else True if tool == OWNER_KNOWLEDGE_TOOL_ID else
                            valid_clock_observation(state.actor, args, observation, context=execution,
                                                    allow_expired=historical_schedule_clock)
                            if tool == OWNER_CLOCK_TOOL_ID else
                            valid_reminder_observation(state.actor, args, observation, context=execution)
                            if tool == OWNER_REMINDER_CREATE_TOOL_ID else
                            (valid_cancel_observation if args.get('operation') == 'cancel' else valid_schedule_observation)(
                                state.actor, args, observation, context=execution)
                            if tool == SCHEDULED_TASK_TOOL_ID else
                            valid_owner_gmail(state.actor, args, observation, context=execution)
                            if tool == OWNER_GMAIL_TOOL_ID else
                            valid_schedules_observation(state.actor, args, observation, context=execution)
                            if tool == OWNER_SCHEDULES_TOOL_ID else
                            validate_skill_catalog_observation(state.actor, tool, args, observation, context=execution)
                            if tool == SKILL_CATALOG_TOOL_ID else
                            validate_owner_named_skill(state.actor, args, observation, context=execution)
                            if tool == OWNER_NAMED_SKILL_TOOL_ID else
                            revalidate_conversation_observation(state.actor, tool, args, observation,
                                                               context=execution, allow_excerpt_truncation=True,
                                                               allow_read_chunks=True, allow_search_overflow=True,
                                                               allow_search_pages=True))):
                raise OwnerRecoveryUnavailable('owner_recovery_unavailable')
            recheck()
            reconstructed[ref] = {'actor': state.actor, 'tool_id': tool, 'arguments': args, 'observation': observation}
            if tool == OWNER_KNOWLEDGE_TOOL_ID:
                reconstructed[ref]['knowledge_snapshot_digest'] = item['knowledge_snapshot_digest']
        conversation_read_sequences(state.actor, {ref: (item['tool_id'], item['arguments'], item['observation'])
                                                   for ref, item in reconstructed.items()})
        conversation_search_sequences(state.actor, {ref: (item['tool_id'], item['arguments'], item['observation'])
                                                     for ref, item in reconstructed.items()})
        recheck()
        return saved, reconstructed
    except Exception:
        raise OwnerRecoveryUnavailable('owner_recovery_unavailable') from None
