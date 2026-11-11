"""Isolated read-only single-loop experiment; NOT a registered runtime engine.

Only synthetic, explicitly injected collaborators are suitable today. This
prototype does not implement the production authority, question, delivery,
effect, engine-stamping or transactional-checkpoint contracts. It rejects all
non-read-only tools, and an in-flight operation on restart blocks rather than
replaying it. A trusting or incorrectly implemented host callback is unsafe;
therefore this module must not be wired to household traffic as-is.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable
from uuid import uuid4

from .evidence_projection import EvidenceAccessError, EvidenceSource, project_evidence
from .task_budget import BudgetExhausted, SQLiteTaskBudgetStore, SyntheticEvaluationBudgetStore
from alphonse.agent_v2.core.tools.invocation import ToolInvocationService


@dataclass
class CandidateState:
    task_id: str
    actor: str
    goal_source: str
    evidence_sources: list[str] = field(default_factory=list)
    status: str = 'running'
    response: str = ''
    blocker: str = ''
    inflight: str = ''
    unsupported_retries: int = 0
    recovery_pending: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {'schema_version': 2, **deepcopy(asdict(self))}

    @classmethod
    def restore(cls, snapshot: dict[str, Any]) -> 'CandidateState':
        if not isinstance(snapshot, dict):
            raise ValueError('candidate_checkpoint_invalid')
        version = snapshot.get('schema_version')
        base = {'schema_version', 'task_id', 'actor', 'goal_source', 'evidence_sources', 'status', 'response', 'blocker', 'inflight'}
        if type(version) is not int or version not in (1, 2):
            raise ValueError('candidate_checkpoint_version_unsupported')
        if set(snapshot) != (base if version == 1 else base | {'unsupported_retries', 'recovery_pending'}):
            raise ValueError('candidate_checkpoint_invalid')
        state = cls(**{key: deepcopy(value) for key, value in snapshot.items() if key != 'schema_version'})
        if (not all(isinstance(x, str) and x.strip() for x in (state.task_id, state.actor, state.goal_source))
                or state.status not in {'running', 'completed', 'waiting_user', 'blocked', 'cancelled'}
                or not isinstance(state.evidence_sources, list)
                or any(not isinstance(x, str) or not x.strip() for x in state.evidence_sources)
                or len(set(state.evidence_sources)) != len(state.evidence_sources)
                or any(not isinstance(x, str) for x in (state.response, state.blocker, state.inflight))
                or type(state.unsupported_retries) is not int or not 0 <= state.unsupported_retries <= 2
                or type(state.recovery_pending) is not bool
                or (state.recovery_pending and state.unsupported_retries == 0)):
            raise ValueError('candidate_checkpoint_invalid')
        return state


@dataclass(frozen=True)
class CandidateHost:
    """Injected test host. All access and grounding callbacks fail closed.

    checkpoint and record_evidence must durably persist before returning; the
    host must serialize exclusive task ownership. Their atomicity with budget
    persistence is not implemented. The prototype does not expose a real user
    question or message sink, and does not own any actor authorization policy.
    `choose(text, max_output_tokens)` must honor its cap and include every input
    byte in the reserved bound. This prototype cannot enforce provider limits.
    """
    task: Any
    context: Any
    budgets: SQLiteTaskBudgetStore
    checkpoint: Callable[[dict[str, Any]], None]
    choose: Callable[[str, int], dict[str, Any]]
    authorize_source: Callable[[str, str], bool]
    load_source: Callable[[str, str], str]
    authorize_read: Callable[[str, str, dict[str, Any]], bool]
    record_evidence: Callable[[str, dict[str, Any]], str]
    verify_response: Callable[[str, str, tuple[str, ...]], bool]
    now_ms: Callable[[], int]
    input_reservation: int = 512
    output_reservation: int = 128
    projection_budget: int = 4096
    deadline_checker: Callable[[], bool] | None = None
    # Only the dedicated synthetic comparison store may opt in. Provider usage
    # remains unknown and full reservations are charged; never enable for users.
    synthetic_allow_unreported_usage: bool = False
    remaining_advertised_evidence: Callable[[str, tuple[str, ...]], bool] | None = None


def _save(host: CandidateHost, state: CandidateState) -> None:
    host.checkpoint(state.snapshot())


def _terminal(host: CandidateHost, state: CandidateState, status: str, reason: str = '', response: str = '') -> CandidateState:
    state.status, state.blocker, state.response = status, reason, response
    _save(host, state)
    return state


def _cancelled(host: CandidateHost) -> bool:
    # Missing/erroring cancellation controls cannot authorize dispatch.
    try:
        return host.context.is_cancelled() is not False
    except Exception:
        return True


def _deadline_elapsed(host: CandidateHost) -> bool:
    # Optional deadline controls are deterministic host policy, not model
    # judgments. Missing controls preserve existing prototype behavior; a
    # configured but erroring or ambiguous control fails closed.
    if host.deadline_checker is None:
        return False
    try:
        return host.deadline_checker() is not False
    except Exception:
        return True


def _project(host: CandidateHost, state: CandidateState):
    sources = [EvidenceSource(state.goal_source, 'goal')]
    sources.extend(EvidenceSource(ref, 'observation', required=True) for ref in state.evidence_sources)
    return project_evidence(sources, actor=state.actor, budget=host.projection_budget,
                            authorize=host.authorize_source, load=host.load_source)


def _projection_fingerprint(projection) -> str:
    # Include actual visible bytes and coverage metadata, not just source IDs:
    # authorized sources may change without changing their stable identities.
    payload = {'text': projection.text, 'coverage': projection.manifest.get('coverage')}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def _is_fresh(host: CandidateHost, state: CandidateState, expected: str) -> bool:
    try:
        return _projection_fingerprint(_project(host, state)) == expected
    except Exception:
        return False


def run_candidate(state: CandidateState, host: CandidateHost) -> CandidateState:
    """Run a single task loop over synthetic read-only tool calls and answers.

    Model usage is settled per call; a missing usage value remains unknown in
    the budget ledger. Failed/incomplete tool outputs are not answer evidence.
    On exhaustion the prototype returns a deterministic blocked state, without
    fabricating completion. Host integrations remain deliberately out of scope.
    """
    if state.status != 'running':
        return state
    if (type(host.synthetic_allow_unreported_usage) is not bool or
            (host.synthetic_allow_unreported_usage and type(host.budgets) is not SyntheticEvaluationBudgetStore)):
        return _terminal(host, state, 'blocked', 'candidate_synthetic_usage_policy_invalid')
    if (not state.task_id or not state.actor or state.actor != getattr(host.task, 'user', None)
            or state.task_id != getattr(host.task, 'task_id', None)):
        return _terminal(host, state, 'blocked', 'candidate_actor_mismatch')
    if state.inflight:
        return _terminal(host, state, 'blocked', 'candidate_unresolved_inflight')
    if _cancelled(host):
        return _terminal(host, state, 'cancelled', 'cancelled')
    while state.status == 'running':
        try:
            projection = _project(host, state)
            fingerprint = _projection_fingerprint(projection)
            # A chooser that adds framing must declare a full-input cap no
            # larger than the durable input reservation. This is only a local
            # accounting check; external transport enforcement is separate.
            chooser_cap = getattr(host.choose, 'max_input_tokens', None)
            if chooser_cap is not None and (type(chooser_cap) is not int or chooser_cap <= 0
                                            or chooser_cap > host.input_reservation):
                return _terminal(host, state, 'blocked', 'candidate_chooser_input_cap_exceeds_reservation')
            # The projection length is only part of the input. A chooser may
            # add framing, tool schemas or other material; its complete cap and
            # the output cap must be enforced by the external transport.
            decision_input = projection.text
            if state.recovery_pending:
                decision_input += ('\nThe proposed answer is not yet verified. Inspect remaining available evidence '
                                   'using authorized tools before answering, or explain what prevents progress.')
            if len(decision_input.encode('utf-8')) > host.input_reservation:
                return _terminal(host, state, 'blocked', 'candidate_input_reservation_too_small')
            budget = host.budgets.load(state.task_id)
            if budget.usage()['in_flight']:
                return _terminal(host, state, 'blocked', 'candidate_unresolved_reservation')
            call_id = 'model-' + uuid4().hex
            revision = budget.revision
            host.budgets.reserve(state.task_id, expected_revision=revision, reservation_id=call_id,
                kind='model', amounts={'actions': 0, 'model_calls': 1, 'input_tokens': host.input_reservation,
                'output_tokens': host.output_reservation}, now_ms=host.now_ms())
            state.inflight = call_id
            _save(host, state)
        except (BudgetExhausted, EvidenceAccessError, ValueError, KeyError) as exc:
            return _terminal(host, state, 'blocked', type(exc).__name__ + ':' + str(exc))
        if _cancelled(host):
            # Proven undispatched here; retain the reservation conservatively.
            return _terminal(host, state, 'cancelled', 'cancelled')
        if _deadline_elapsed(host):
            return _terminal(host, state, 'blocked', 'candidate_deadline_exceeded')
        try:
            decision = host.choose(decision_input, host.output_reservation)
        except Exception:
            decision = None
        budget = host.budgets.load(state.task_id)
        revision = budget.revision
        usage = decision.get('usage', {}) if isinstance(decision, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        try:
            budget.settle(call_id, outcome='success' if isinstance(decision, dict) else 'failure',
                          input_tokens=usage.get('input_tokens'), output_tokens=usage.get('output_tokens'))
            host.budgets.save(state.task_id, budget, expected_revision=revision)
        except (ValueError, KeyError):
            return _terminal(host, state, 'blocked', 'candidate_model_settlement_unavailable')
        state.inflight = ''
        _save(host, state)
        if _cancelled(host):
            return _terminal(host, state, 'cancelled', 'cancelled')
        if not isinstance(decision, dict):
            return _terminal(host, state, 'blocked', 'candidate_model_unavailable')
        if any((value is not None and (type(value) is not int or value < 0 or value > limit))
               or (value is None and not host.synthetic_allow_unreported_usage)
               for value, limit in ((usage.get('input_tokens'), host.input_reservation),
                                    (usage.get('output_tokens'), host.output_reservation))):
            return _terminal(host, state, 'blocked', 'candidate_model_usage_unknown_or_overrun')
        # A still-authorized source may nevertheless have changed during inference.
        if not _is_fresh(host, state, fingerprint):
            return _terminal(host, state, 'blocked', 'candidate_evidence_changed_or_revoked')
        kind = decision.get('kind')
        if kind in {'answer', 'question'}:
            text = decision.get('text')
            if not isinstance(text, str) or not text.strip():
                return _terminal(host, state, 'blocked', 'candidate_empty_communication')
            if kind == 'question':
                if _deadline_elapsed(host):
                    return _terminal(host, state, 'blocked', 'candidate_deadline_exceeded')
                return _terminal(host, state, 'waiting_user', 'question_delivery_not_integrated', text)
            try:
                # Re-read under the current actor before accepting any response.
                grounded = host.verify_response(state.actor, text, tuple(state.evidence_sources)) is True
            except Exception:
                return _terminal(host, state, 'blocked', 'candidate_verification_unavailable')
            if _cancelled(host):
                return _terminal(host, state, 'cancelled', 'cancelled')
            if not _is_fresh(host, state, fingerprint):
                return _terminal(host, state, 'blocked', 'candidate_evidence_changed_or_revoked')
            if _cancelled(host):
                return _terminal(host, state, 'cancelled', 'cancelled')
            if _deadline_elapsed(host):
                return _terminal(host, state, 'blocked', 'candidate_deadline_exceeded')
            if not grounded:
                missing = False
                if host.remaining_advertised_evidence is not None and state.unsupported_retries < 2:
                    try:
                        missing = host.remaining_advertised_evidence(state.actor, tuple(state.evidence_sources)) is True
                    except Exception:
                        missing = False
                if missing:
                    state.unsupported_retries += 1
                    state.recovery_pending = True
                    _save(host, state)
                    continue
                return _terminal(host, state, 'blocked', 'candidate_answer_unsupported')
            return _terminal(host, state, 'completed', response=text)
        if kind != 'tool' or not isinstance(decision.get('tool_id'), str) or not isinstance(decision.get('arguments'), dict):
            return _terminal(host, state, 'blocked', 'candidate_decision_invalid')
        tool_id, args = decision['tool_id'], deepcopy(decision['arguments'])
        registry = getattr(host.context, 'tools', None)
        descriptor = registry.get(tool_id) if registry is not None else None
        if descriptor is None or getattr(getattr(descriptor, 'kind', None), 'value', '') != 'native' or descriptor.read_only is not True:
            return _terminal(host, state, 'blocked', 'candidate_requires_read_only_native_tool')
        try:
            permitted = host.authorize_read(state.actor, tool_id, args) is True
        except Exception:
            permitted = False
        if not permitted:
            return _terminal(host, state, 'blocked', 'candidate_read_denied')
        if _cancelled(host):
            return _terminal(host, state, 'cancelled', 'cancelled')
        action_id = 'tool-' + uuid4().hex
        try:
            budget = host.budgets.load(state.task_id)
            revision = budget.revision
            host.budgets.reserve(state.task_id, expected_revision=revision, reservation_id=action_id,
                kind='tool', amounts={'actions': 1, 'model_calls': 0, 'input_tokens': 0, 'output_tokens': 0},
                now_ms=host.now_ms())
            state.inflight = action_id
            _save(host, state)
        except (BudgetExhausted, ValueError, KeyError) as exc:
            return _terminal(host, state, 'blocked', type(exc).__name__ + ':' + str(exc))
        if _cancelled(host):
            return _terminal(host, state, 'cancelled', 'cancelled')
        # Re-check revocation at dispatch; native tool invocation applies its own
        # validation/context but does not substitute for host authorization.
        try:
            permitted = host.authorize_read(state.actor, tool_id, args) is True
        except Exception:
            permitted = False
        if not permitted:
            return _terminal(host, state, 'blocked', 'candidate_read_denied')
        if not _is_fresh(host, state, fingerprint):
            return _terminal(host, state, 'blocked', 'candidate_evidence_changed_or_revoked')
        if _cancelled(host):
            return _terminal(host, state, 'cancelled', 'cancelled')
        if _deadline_elapsed(host):
            return _terminal(host, state, 'blocked', 'candidate_deadline_exceeded')
        result = ToolInvocationService(context=host.context, task=host.task).invoke(tool_id, args, call_id=action_id)
        budget = host.budgets.load(state.task_id)
        revision = budget.revision
        budget.settle(action_id, outcome='success' if result.get('status') == 'success' else 'failure')
        host.budgets.save(state.task_id, budget, expected_revision=revision)
        # Retain the in-flight marker until the verified observation and its
        # source reference are checkpointed. A crash after settlement must not
        # silently repeat the read or discard its result on resume.
        if _cancelled(host):
            return _terminal(host, state, 'cancelled', 'cancelled')
        try:
            permitted = host.authorize_read(state.actor, tool_id, args) is True
        except Exception:
            permitted = False
        if not permitted:
            return _terminal(host, state, 'blocked', 'candidate_read_revoked')
        if not _is_fresh(host, state, fingerprint):
            return _terminal(host, state, 'blocked', 'candidate_evidence_changed_or_revoked')
        if result.get('status') != 'success':
            return _terminal(host, state, 'blocked', 'candidate_tool_failed_or_waiting')
        payload = result.get('result')
        # Tool-specific coverage/partial semantics must be verified by the host;
        # this prototype only accepts a deliberately explicit fixture contract.
        if not isinstance(payload, dict) or payload.get('candidate_complete') is not True:
            return _terminal(host, state, 'blocked', 'candidate_observation_incomplete')
        ref = host.record_evidence(state.actor, result)
        if not isinstance(ref, str) or not ref.strip() or ref == state.goal_source or ref in state.evidence_sources:
            return _terminal(host, state, 'blocked', 'candidate_evidence_reference_invalid')
        state.evidence_sources.append(ref)
        state.recovery_pending = False
        state.inflight = ''
        _save(host, state)
    return state
