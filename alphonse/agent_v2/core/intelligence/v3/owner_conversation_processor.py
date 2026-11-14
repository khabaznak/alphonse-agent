"""Disposable CoreLoop bridge for the experimental owner conversation path.

This deliberately accepts only an in-memory queue and the synthetic budget
store. It cannot authorize household traffic, provide durable restart/replay
protection, deliver an answer, or enable a production engine. The actual core
and router can exercise a fixture host through this adapter while those missing
boundaries are built. A generated draft is not a delivered response.

CoreLoop's WAITING status is not a publication or side-effect fence: the core
can finish memory and acknowledge consumed messages before returning it. Never
attach a real memory, queue, conversation store, activity or publication sink
to this fixture. The adapter rejects several such collaborators, but CoreLoop
wraps its activity sink even when none is configured, and these checks happen
after CoreLoop prework. They do not protect an incorrectly configured live core.
"""
from __future__ import annotations

from threading import Lock
from typing import Callable

from alphonse.agent_v2.core.core import ProcessingResult, ProcessingStatus, StateSnapshot
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from .single_loop_candidate import CandidateHost, CandidateState, run_candidate
from .task_budget import SyntheticEvaluationBudgetStore


class SyntheticOwnerConversationProcessor:
    """Explicitly injected, single-process fixture adapter, never a live host."""

    def __init__(self, *, owner: str, host_factory: Callable):
        if not isinstance(owner, str) or not owner.strip() or not callable(host_factory):
            raise ValueError('synthetic_owner_configuration_invalid')
        self._owner = owner
        self._host_factory = host_factory
        self._seen_messages: set[str] = set()
        self._lock = Lock()

    def process(self, task, context):
        # A task's user field alone is not production authentication. These
        # checks only scope an explicitly injected, disposable fixture.
        if (type(context.messages) is not InMemoryMessageQueue
                or context.memory is not None or context.conversation_store is not None
                or any(sink is not None for sink in (context.ui_event_sink, context.telemetry_sink,
                                                      context.delivery_sink))
                or task.user != self._owner or not isinstance(task.message_id, str)
                or not task.message_id.strip() or not isinstance(task.task_id, str)
                or not task.task_id.strip()):
            return self._failure(task, 'synthetic_owner_context_unavailable')
        with self._lock:
            if task.message_id in self._seen_messages:
                return self._failure(task, 'synthetic_owner_replay_unavailable')
            self._seen_messages.add(task.message_id)
        try:
            host, state = self._host_factory(task, context)
            if (not isinstance(host, CandidateHost) or type(host.budgets) is not SyntheticEvaluationBudgetStore
                    or host.task is not task or host.context is not context
                    or not isinstance(state, CandidateState) or state.task_id != task.task_id
                    or state.actor != task.user or state.status != 'running' or state.inflight):
                return self._failure(task, 'synthetic_owner_host_invalid')
            result = run_candidate(state, host)
        except Exception:
            # Do not leak potentially sensitive provider/host exception content.
            return self._failure(task, 'synthetic_owner_execution_unavailable')
        if result.status == 'cancelled':
            return ProcessingResult(snapshot=StateSnapshot(task_owner=task.user,
                metadata={'engine': 'synthetic_owner_candidate', 'status': 'cancelled'}),
                status=ProcessingStatus.CANCELLED, error='synthetic_owner_cancelled')
        if result.status != 'completed':
            return self._failure(task, 'synthetic_owner_candidate_blocked')
        # Return a fixture-only draft. The normal response/question and outbox
        # paths are not connected, so no delivery or task completion is claimed.
        return ProcessingResult(snapshot=StateSnapshot(task_owner=task.user, current_work=task.goal,
            metadata={'engine': 'synthetic_owner_candidate', 'status': 'draft_only',
                      'draft': result.response, 'delivery': 'not_integrated'}),
            status=ProcessingStatus.WAITING)

    @staticmethod
    def _failure(task, reason):
        return ProcessingResult(snapshot=StateSnapshot(task_owner=task.user,
            metadata={'engine': 'synthetic_owner_candidate', 'status': 'blocked', 'reason': reason}),
            status=ProcessingStatus.FAILED, error=reason)
