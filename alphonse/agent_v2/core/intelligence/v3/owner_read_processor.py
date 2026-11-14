"""Direct-owner conversation-read candidate processor, not registered for live use.

The caller must supply a trusted, claim-aware authorization callback and a
durable checkpoint writer. This processor makes real actor-scoped native reads,
but its budget, checkpoint and queue publication are not one fenced
transaction. It returns only an undelivered draft; do not enable it for the
daemon until trusted ingress, durable ownership and delivery are integrated.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import time
from typing import Any, Callable
from uuid import uuid4

from alphonse.agent_v2.core.core import CoreLoopContext, ProcessingResult, ProcessingStatus, StateSnapshot
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.tools.registry.native.memory import (
    READ_MEMORY_SOURCE_TOOL_ID, SEARCH_MEMORY_TOOL_ID, revalidate_conversation_observation,
)
from alphonse.agent_v2.users import V2UserStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from .router_candidate_chooser import RouterCandidateChooser
from .single_loop_candidate import CandidateHost, CandidateState, run_candidate
from .task_budget import BudgetLimits, SQLiteTaskBudgetStore, TaskBudget
from .single_loop_checkpoint import ExperimentalCheckpointStore
from .owner_atomic_checkpoint import OwnerAtomicCheckpoint
from .owner_read_recovery import restore_owner_read_observations
from datetime import datetime, timezone


def authorize_trusted_owner_claim(task: TaskState, context: CoreLoopContext) -> bool:
    """Check the acquired receipt against trusted ingress and the current lease.

    This is point-in-time authorization, not a fence around subsequent effects.
    The daemon and core must remain disabled until those effects are fenced.
    """
    return (type(context.messages) is SQLiteMessageQueue
            and context.messages.verify_trusted_local_owner_claim(context.claim_receipt, task))


class OwnerReadConversationProcessor:
    """Run the real read-only candidate for a trusted active direct owner.

    `authorize_claim` is a trusted runtime boundary, not model output or task
    metadata. A true result must establish the current actor and exclusive
    queue claim. The present daemon does not supply that boundary, so this
    processor is deliberately not registered. `save_checkpoint` must persist
    each snapshot before returning. Drafts and questions are never delivered.
    """

    def __init__(self, *, users: V2UserStore, budgets: SQLiteTaskBudgetStore,
                 authorize_claim: Callable[[TaskState, CoreLoopContext], bool],
                 save_checkpoint: Callable[[TaskState, dict[str, Any]], None],
                 now_ms: Callable[[], int] | None = None,
                 atomic_checkpoints: ExperimentalCheckpointStore | None = None,
                 selected_draft_sink: Callable[[OwnerAtomicCheckpoint, CoreLoopContext], None] | None = None):
        if (not isinstance(users, V2UserStore) or type(budgets) is not SQLiteTaskBudgetStore
                or not callable(authorize_claim) or not callable(save_checkpoint)):
            raise ValueError('owner_read_configuration_invalid')
        self.users, self.budgets = users, budgets
        self.authorize_claim, self.save_checkpoint = authorize_claim, save_checkpoint
        self.now_ms = now_ms or (lambda: int(time.time() * 1000))
        if atomic_checkpoints is not None and not isinstance(atomic_checkpoints, ExperimentalCheckpointStore):
            raise ValueError('owner_read_atomic_checkpoint_invalid')
        if selected_draft_sink is not None and (atomic_checkpoints is None or not callable(selected_draft_sink)):
            raise ValueError('owner_read_draft_sink_invalid')
        self.atomic_checkpoints = atomic_checkpoints
        self.selected_draft_sink = selected_draft_sink

    def _authorized(self, task: TaskState, context: CoreLoopContext) -> bool:
        try:
            user = self.users.get_user(str(task.user or ''))
            return bool(user is not None and user.is_active is True and user.role == 'admin'
                        and self.users.is_admin(user.user_id)
                        and user.user_id == task.user and self.authorize_claim(task, context) is True
                        and context.is_cancelled() is False)
        except Exception:
            return False

    def process(self, task: TaskState, context: CoreLoopContext) -> ProcessingResult:
        if (not isinstance(task, TaskState) or not isinstance(context, CoreLoopContext)
                or task.intelligence_engine != 'single_loop_experimental'
                or task.intelligence_schema_version != 1 or not task.message_id
                or task.task_id != task.message_id or not self._authorized(task, context)):
            return self._failure(task, 'owner_read_authority_unavailable')
        actor = task.user
        if context.inference is None or context.tools is None or context.conversation_store is None:
            return self._failure(task, 'owner_read_host_unavailable')
        descriptors = tuple(context.tools.get(key) for key in (SEARCH_MEMORY_TOOL_ID, READ_MEMORY_SOURCE_TOOL_ID))
        if any(item is None or item.read_only is not True for item in descriptors):
            return self._failure(task, 'owner_read_tools_unavailable')
        # Advertise the exact narrower scope the host will permit, so the
        # chooser can form an executable read without learning private data.
        try:
            search_schema = deepcopy(descriptors[0].argument_schema)
            search_schema['properties']['source_types'] = {
                'type': 'array', 'items': {'type': 'string', 'enum': ['conversation']},
                'minItems': 1, 'maxItems': 1,
            }
            search_schema['required'] = list(dict.fromkeys([*search_schema.get('required', []), 'source_types']))
            descriptors = (
                replace(descriptors[0], argument_schema=search_schema,
                        description='Search the owner’s prior conversation messages. Results are historical, not proof of current state.'),
                replace(descriptors[1], description='Read an owner conversation source ID returned by search; historical messages may be incomplete or outdated.'),
            )
            chooser = RouterCandidateChooser(router=context.inference, task=task, tools=descriptors,
                max_input_tokens=65536, cancellation_checker=context.is_cancelled)
            vector = lambda a, m, i, o: {'actions': a, 'model_calls': m, 'input_tokens': i, 'output_tokens': o}
            budget = TaskBudget(BudgetLimits(vector(5, 8, 524288, 131072),
                vector(0, 1, 65536, 16384), 120000, 5000), started_at_ms=self.now_ms())
            atomic = None
            candidate_state = CandidateState(task.task_id, actor, 'goal')
            restored_observations = {}
            if self.atomic_checkpoints is None:
                # Existing task IDs fail closed: no blind replay after restart.
                self.budgets.create(task.task_id, budget)
            else:
                atomic = OwnerAtomicCheckpoint(context.messages, self.atomic_checkpoints, task,
                    context.claim_receipt, clock=lambda: datetime.now(timezone.utc))
                version = getattr(task, '_checkpoint_version', None)
                if version is None:
                    atomic.initialize(candidate_state, budget)
                else:
                    recovered, restored_observations = restore_owner_read_observations(
                        atomic, context, expected_version=version)
                    candidate_state = recovered.state
                    atomic.version = recovered.version
        except Exception:
            return self._failure(task, 'owner_read_initialization_unavailable')

        # Each recovered binding has already been checked against the current
        # source and is checked again at each later access.
        observations: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {
            ref: (item['tool_id'], deepcopy(item['arguments']), deepcopy(item['observation']))
            for ref, item in restored_observations.items()
        }
        execution_context = context.tool_execution_context(task)

        def allowed(actor_id: str, tool_id: str, args: dict[str, Any]) -> bool:
            if actor_id != actor or not self._authorized(task, context) or not isinstance(args, dict):
                return False
            if tool_id == SEARCH_MEMORY_TOOL_ID:
                return args.get('source_types') == ['conversation'] and args.get('offset', 0) == 0
            return (tool_id == READ_MEMORY_SOURCE_TOOL_ID and isinstance(args.get('source_id'), str)
                    and args['source_id'].startswith('conversation:') and args.get('offset', 0) == 0)

        def valid(actor_id: str, tool_id: str, args: dict[str, Any], payload: dict[str, Any]) -> bool:
            return allowed(actor_id, tool_id, args) and revalidate_conversation_observation(
                actor_id, tool_id, args, payload, context=execution_context)

        def access(actor_id: str, ref: str) -> bool:
            if actor_id != actor or not self._authorized(task, context):
                return False
            if ref == 'goal':
                return True
            bound = observations.get(ref)
            return bool(bound and valid(actor_id, bound[0], bound[1], bound[2]))

        def load(actor_id: str, ref: str) -> str:
            if not access(actor_id, ref):
                raise ValueError('owner_read_evidence_unavailable')
            return str(task.goal or '') if ref == 'goal' else json.dumps(observations[ref][2], ensure_ascii=False, sort_keys=True)

        def record(actor_id: str, tool_id: str, args: dict[str, Any], result: dict[str, Any]) -> str:
            payload = result.get('result') if isinstance(result, dict) else None
            if not isinstance(payload, dict) or not valid(actor_id, tool_id, args, payload):
                raise ValueError('owner_read_evidence_unavailable')
            ref = 'owner-read:' + uuid4().hex
            observations[ref] = (tool_id, deepcopy(args), deepcopy(payload))
            return ref

        def validate(actor_id: str, tool_id: str, args: dict[str, Any], payload: dict[str, Any]) -> bool:
            return valid(actor_id, tool_id, args, payload)

        # This verifies only retained source access/provenance, never the
        # semantic truth of free-form model output. The result remains a draft.
        def provenance(actor_id: str, text: str, refs: tuple[str, ...]) -> bool:
            return bool(text.strip() and self._authorized(task, context) and all(access(actor_id, ref) for ref in refs))

        host = CandidateHost(task, context, self.budgets, lambda snapshot: self.save_checkpoint(task, snapshot),
            chooser, access, load, allowed, lambda *_: '', provenance, self.now_ms,
            input_reservation=65536, output_reservation=16384, projection_budget=65536,
            validate_observation=validate, record_bound_observation=record, atomic_checkpoint=atomic)
        try:
            result = run_candidate(candidate_state, host)
        except Exception:
            return self._failure(task, 'owner_read_execution_unavailable')
        if result.status == 'cancelled':
            return ProcessingResult(snapshot=StateSnapshot(task_owner=actor, metadata={
                                        'engine': 'single_loop_experimental',
                                        'intelligence_schema_version': 1, 'status': 'cancelled'}),
                                    status=ProcessingStatus.CANCELLED, error='owner_read_cancelled')
        if result.status not in {'completed', 'waiting_user'}:
            return self._failure(task, result.blocker or 'owner_read_blocked')
        # Owner pickup currently supports answers only. Questions stay as
        # undelivered drafts until a claim-aware question lifecycle exists.
        if result.status == 'completed' and self.selected_draft_sink is not None:
            try:
                # The callback receives the server-owned checkpoint object,
                # never a caller-supplied receipt or answer. It must itself
                # revalidate current authority before any later publication.
                self.selected_draft_sink(atomic, context)
            except Exception:
                return self._failure(task, 'owner_read_draft_handoff_unavailable')
        return ProcessingResult(snapshot=StateSnapshot(task_owner=actor, current_work=task.goal,
            metadata={'engine': 'single_loop_experimental', 'intelligence_schema_version': 1,
                      'status': 'draft_only', 'draft': result.response, 'delivery': 'not_integrated'}),
            status=ProcessingStatus.WAITING)

    @staticmethod
    def _failure(task: Any, reason: str) -> ProcessingResult:
        return ProcessingResult(snapshot=StateSnapshot(task_owner=getattr(task, 'user', None),
            metadata={'engine': 'single_loop_experimental', 'intelligence_schema_version': 1,
                      'status': 'blocked', 'reason': reason}), status=ProcessingStatus.FAILED, error=reason)
