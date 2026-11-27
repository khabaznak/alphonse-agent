"""Simple-loop candidate with narrowly scoped injected host controls.

Ordinary read paths and the owner reminder have separate production host
guards. A disposable evaluation host can explicitly supply a Bash callback;
it must enforce isolation and is not a household effect boundary.
An in-flight operation on restart blocks rather than replaying it.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable
from uuid import uuid4

from alphonse.agent_v2.core.inference.models import ModelResponseParseError
from alphonse.agent_v2.core.intelligence.context_projection import ContextBudgetExceeded
from .evidence_projection import EvidenceAccessError, EvidenceSource, project_evidence
from .task_budget import BudgetExhausted, SQLiteTaskBudgetStore, SyntheticEvaluationBudgetStore
from .prerequisite import FEEDBACK, MissingPrerequisite
from .owner_disposable_handoff import OWNER_DISPOSABLE_HANDOFF_TOOL_ID
from alphonse.agent_v2.core.tools.invocation import ToolInvocationService


logger = logging.getLogger(__name__)


class ClarificationReviewUnavailable(Exception):
    """The clarification reviewer could not be reached or configured."""


_CHOOSER_FAILURES = {
    'candidate_router_input_exceeds_reservation': 'input_exceeds_reservation',
    'candidate_router_output_exceeds_reservation': 'output_exceeds_reservation',
    'candidate_router_source_progress_unavailable': 'source_progress_unavailable',
    'candidate_router_result_invalid': 'result_invalid',
    'candidate_router_cancelled': 'cancelled',
    'candidate_router_unavailable': 'provider_unavailable',
    **{'candidate_router_provider_' + category: 'provider_' + category for category in (
        'model_not_configured', 'cli_missing', 'subscription_auth_unverified',
        'authentication_required', 'cli_upgrade_required', 'provider_timeout',
        'empty_response', 'invalid_response', 'cancelled')},
}


def _chooser_failure_category(exc: Exception) -> str:
    if type(exc) is ValueError and len(exc.args) == 1 and type(exc.args[0]) is str:
        return _CHOOSER_FAILURES.get(exc.args[0], 'chooser_error')
    return 'chooser_error'


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
    invalid_decision_retries: int = 0
    decision_format_pending: bool = False
    answer_review_state: str = 'not_started'
    prerequisite_feedback: str = ''
    prerequisite_retries: int = 0
    retry_feedback_pending: bool = False
    parse_repair_pending: bool = False
    # Preserve progress metadata in checkpoints written by development builds.
    progress_reviewed_count: int = 0
    progress_assessment: str = 'unknown'
    reused_web_fetch_ref: str = ''

    def request_parse_repair(self, budget) -> bool:
        if self.invalid_decision_retries or budget.usage()['unknown_usage'] or not budget.within_work_limits():
            return False
        self.invalid_decision_retries = 1
        self.decision_format_pending = True
        self.parse_repair_pending = True
        return True

    def snapshot(self) -> dict[str, Any]:
        return {'schema_version': 9, **deepcopy(asdict(self))}

    @classmethod
    def restore(cls, snapshot: dict[str, Any]) -> 'CandidateState':
        if not isinstance(snapshot, dict):
            raise ValueError('candidate_checkpoint_invalid')
        version = snapshot.get('schema_version')
        base = {'schema_version', 'task_id', 'actor', 'goal_source', 'evidence_sources', 'status', 'response', 'blocker', 'inflight'}
        if type(version) is not int or version not in (1, 2, 3, 4, 5, 6, 7, 8, 9):
            raise ValueError('candidate_checkpoint_version_unsupported')
        expected = base if version == 1 else base | {'unsupported_retries', 'recovery_pending'}
        if version >= 3:
            expected |= {'invalid_decision_retries', 'decision_format_pending'}
        if version >= 4:
            expected |= {'answer_review_state'}
        if version >= 5:
            expected |= {'prerequisite_feedback', 'prerequisite_retries'}
        if version >= 6:
            expected |= {'retry_feedback_pending'}
        if version >= 7:
            expected |= {'parse_repair_pending'}
        if version >= 8:
            expected |= {'progress_reviewed_count', 'progress_assessment'}
        if version >= 9:
            expected |= {'reused_web_fetch_ref'}
        if set(snapshot) != expected:
            raise ValueError('candidate_checkpoint_invalid')
        state = cls(**{key: deepcopy(value) for key, value in snapshot.items() if key != 'schema_version'})
        if (not all(isinstance(x, str) and x.strip() for x in (state.task_id, state.actor, state.goal_source))
                or state.status not in {'running', 'completed', 'waiting_user', 'blocked', 'cancelled'}
                or not isinstance(state.evidence_sources, list)
                or any(not isinstance(x, str) or not x.strip() for x in state.evidence_sources)
                or len(set(state.evidence_sources)) != len(state.evidence_sources)
                or type(state.progress_reviewed_count) is not int
                or not 0 <= state.progress_reviewed_count <= len(state.evidence_sources)
                or state.progress_assessment not in ('progress', 'no_progress', 'unknown')
                or any(not isinstance(x, str) for x in (state.response, state.blocker, state.inflight))
                or type(state.unsupported_retries) is not int or not 0 <= state.unsupported_retries <= 2
                or type(state.recovery_pending) is not bool
                or (state.recovery_pending and state.unsupported_retries == 0)
                or type(state.invalid_decision_retries) is not int or not 0 <= state.invalid_decision_retries <= 1
                or type(state.decision_format_pending) is not bool
                or (state.decision_format_pending and state.invalid_decision_retries == 0)
                or type(state.parse_repair_pending) is not bool
                or (state.parse_repair_pending and not state.decision_format_pending)
                or not isinstance(state.answer_review_state, str)
                or state.answer_review_state not in {'not_started', 'pending', 'used'}):
            raise ValueError('candidate_checkpoint_invalid')
        if (state.prerequisite_feedback not in {'', *FEEDBACK}
                or type(state.prerequisite_retries) is not int or not 0 <= state.prerequisite_retries <= 2
                or (state.prerequisite_feedback and not state.prerequisite_retries)
                or type(state.retry_feedback_pending) is not bool
                or not isinstance(state.reused_web_fetch_ref, str)
                or (state.reused_web_fetch_ref and state.reused_web_fetch_ref not in state.evidence_sources)):
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
    # Give the owner host one budgeted chance to reconsider a rejected answer.
    # This never changes the verification, authority, cancellation or effect gates.
    retry_unverified_answer: bool = False
    # Evaluation-only self-review, not independent verification or authority.
    synthetic_preanswer_review: bool = False
    # A real read may have an operation-specific completion contract instead
    # of the synthetic candidate_complete marker. This only validates the
    # observation shape; it is not authority or a publication fence.
    validate_observation: Callable[[str, str, dict[str, Any], dict[str, Any]], bool] | None = None
    # Optional action-bound recording for real read hosts. It must persist the
    # actor, tool, exact arguments and result before returning the evidence ref.
    # A ref alone does not authorize reuse or prove the observation is current.
    record_bound_observation: Callable[[str, str, dict[str, Any], dict[str, Any]], str] | None = None
    # Experimental shared-claim checkpoint protocol. Default execution keeps
    # the existing budget/checkpoint behavior and is not production enabled.
    atomic_checkpoint: Any | None = None
    # Narrow host-owned local mutation. The callback atomically commits the
    # effect, budget and terminal checkpoint; it never dispatches generic tools.
    commit_owner_reminder: Callable[[Any, dict, str], Any] | None = None
    commit_owner_schedule: Callable[[Any, dict, str], Any] | None = None
    commit_disposable_handoff: Callable[[Any, dict, str], Any] | None = None
    allow_question: Callable[[str], bool] | None = None
    # Trusted owner host only. Neither callback is supplied by the model.
    read_private_knowledge: Callable[[], Any] | None = None
    record_private_knowledge: Callable[[str, str, dict, dict, str], str] | None = None
    # Disposable evaluation host only; these callbacks grant no household access.
    # Successful invocation conveys an observation, not verified task success;
    # a completed nonzero command can be observed for model recovery. Callback
    # exceptions preserve unknown in-flight work. The callback must enforce its
    # own cancellation, isolation, cleanup, bounded output and timeout controls.
    authorize_disposable_tool: Callable[[str, str, dict[str, Any]], bool] | None = None
    execute_disposable_tool: Callable[[str, str, dict[str, Any]], dict[str, Any]] | None = None
    # Non-persistent Dojo session only. Never enable for a household shell.
    disposable_atomic_session: str = ''
    # Explicit trusted-owner host adapter. A returned process is evidence of
    # process output, not verification that its requested effects occurred.
    authorize_host_bash: Callable[[str, str, dict[str, Any]], bool] | None = None
    execute_host_bash: Callable[[str, dict[str, Any], str], dict[str, Any]] | None = None
    defer_identical_failed_read: Callable[[str, str, dict[str, Any], tuple[str, ...]], bool] | None = None
    covered_web_fetch: Callable[[str, str, dict[str, Any], tuple[str, ...]], str | None] | None = None
    # The host may durably pause at a quiescent boundary. It must persist and
    # fence the pause before returning True; this callback grants no authority.
    checkpoint_pause: Callable[[CandidateState], bool] | None = None
    prepare_choice: Callable[[str, int], Callable[[], dict]] | None = None
    compact_source: Callable[[str, str, int], str] | None = None
    review_progress_checkpoint: Callable[[CandidateState], str] | None = None


def _save(host: CandidateHost, state: CandidateState, *, consume_decision: bool = False,
          selected_decision: dict | None = None, projection_fingerprint: str = '') -> None:
    if host.atomic_checkpoint is not None:
        host.atomic_checkpoint.save_state(state, consume_decision=consume_decision,
            selected_decision=selected_decision, projection_fingerprint=projection_fingerprint)
    else:
        host.checkpoint(state.snapshot())


def _terminal(host: CandidateHost, state: CandidateState, status: str, reason: str = '', response: str = '', *,
              selected_decision: dict | None = None, projection_fingerprint: str = '') -> CandidateState:
    state.status, state.blocker, state.response = status, reason, response
    _save(host, state, consume_decision=True, selected_decision=selected_decision,
          projection_fingerprint=projection_fingerprint)
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


def _mark_steering_included(host: CandidateHost, decision: dict, fingerprint: str) -> bool:
    if host.atomic_checkpoint is None:
        return True
    try:
        host.atomic_checkpoint.mark_steering_included(decision, fingerprint, host.context.cancellation_checker)
        return True
    except Exception:
        return False


def _project(host: CandidateHost, state: CandidateState, compact_limit: int = 0):
    sources = [EvidenceSource(state.goal_source, 'goal')]
    sources.extend(EvidenceSource(ref, 'observation', required=True) for ref in state.evidence_sources)
    def compact(ref, content):
        # Preserve the requested page when other observations can be shortened
        # enough. The smallest tier remains available for large pages.
        if ref == state.reused_web_fetch_ref and compact_limit > 512:
            return content
        return host.compact_source(ref, content, compact_limit)
    return project_evidence(sources, actor=state.actor, budget=host.projection_budget,
                            authorize=host.authorize_source, load=host.load_source,
                            compact=compact if compact_limit else None)


def _projection_fingerprint(projection) -> str:
    # Include actual visible bytes and coverage metadata, not just source IDs:
    # authorized sources may change without changing their stable identities.
    payload = {'text': projection.text, 'coverage': projection.manifest.get('coverage')}
    if 'compacted_source_digests' in projection.manifest:
        payload['compacted_source_digests'] = projection.manifest['compacted_source_digests']
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def _is_fresh(host: CandidateHost, state: CandidateState, expected: str, compact_limit: int = 0) -> bool:
    try:
        return _projection_fingerprint(_project(host, state, compact_limit)) == expected
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
    if (type(host.synthetic_preanswer_review) is not bool or
            (not host.synthetic_preanswer_review and state.answer_review_state != 'not_started')):
        return _terminal(host, state, 'blocked', 'candidate_answer_review_policy_invalid')
    if (not state.task_id or not state.actor or state.actor != getattr(host.task, 'user', None)
            or state.task_id != getattr(host.task, 'task_id', None)):
        return _terminal(host, state, 'blocked', 'candidate_actor_mismatch')
    if state.inflight:
        return _terminal(host, state, 'blocked', 'candidate_unresolved_inflight')
    if _cancelled(host):
        return _terminal(host, state, 'cancelled', 'cancelled')
    while state.status == 'running':
        # A prior step's checkpoint or an external actor may have cancelled
        # before this iteration. Do not load sources or reserve another model
        # call after cancellation is already observable.
        if _cancelled(host):
            return _terminal(host, state, 'cancelled', 'cancelled')
        if host.checkpoint_pause is not None:
            # An exception must escape rather than dispatching more work or
            # overwriting a pause that may already have been committed.
            paused = host.checkpoint_pause(deepcopy(state))
            if type(paused) is not bool:
                raise ValueError('candidate_pause_result_invalid')
            if paused:
                return state
        repairing_parse = state.parse_repair_pending
        repair_failure = 'candidate_model_response_invalid' if repairing_parse else ''
        if repairing_parse and _deadline_elapsed(host):
            return _terminal(host, state, 'blocked', repair_failure)
        if (host.review_progress_checkpoint is not None and not repairing_parse
                and len(state.evidence_sources) - state.progress_reviewed_count >= 3):
            assessment = host.review_progress_checkpoint(state)
            state.progress_assessment = assessment if assessment in ('progress', 'no_progress') else 'unknown'
            state.progress_reviewed_count = len(state.evidence_sources)
            _save(host, state)
            if _cancelled(host):
                return _terminal(host, state, 'cancelled', 'cancelled')
        try:
            # A chooser that adds framing must declare a full-input cap no
            # larger than the durable input reservation. This is only a local
            # accounting check; external transport enforcement is separate.
            chooser_cap = getattr(host.choose, 'max_input_tokens', None)
            if chooser_cap is not None and (type(chooser_cap) is not int or chooser_cap <= 0
                                            or chooser_cap > host.input_reservation):
                return _terminal(host, state, 'blocked', repair_failure or 'candidate_chooser_input_cap_exceeds_reservation')
            # The projection length is only part of the input. A chooser may
            # add framing, tool schemas or other material; its complete cap and
            # the output cap must be enforced by the external transport.
            decision_input = ''
            if (not repairing_parse and state.progress_assessment == 'no_progress'
                    and state.progress_reviewed_count == len(state.evidence_sources)):
                decision_input += ('\nA checkpoint review found no clear progress. Reassess the approach '
                                   'using the evidence so far. Choose a materially different authorized route if '
                                   'one can resolve an open question, or explain what prevents progress.')
            if state.recovery_pending:
                decision_input += ('\nA previous answer was not verified; the reason may be unknown. '
                                   'Consider relevant authorized sources or another approach. If you cannot '
                                   'make progress, explain the specific limitation or ask for information '
                                   'only the user can supply. Do not claim an unverified effect.')
            if state.prerequisite_feedback:
                decision_input += '\n' + FEEDBACK[state.prerequisite_feedback]
            if state.retry_feedback_pending:
                decision_input += ('\nThe previous identical read failed, so its immediate repeat was not dispatched. '
                                   'Consider another available route or explain what remains unavailable.')
            if state.reused_web_fetch_ref:
                decision_input += (f'\nA requested page is already recorded at {state.reused_web_fetch_ref}. '
                                   'Use the recorded text. If it leaves a material gap, choose a different source '
                                   'or explain what remains uncertain. Repeating the same fetch cannot add evidence.')
            if repairing_parse:
                decision_input += ('\nThe previous model response could not be decoded. Return one valid answer or '
                                   'question using the recorded observations. Explain any limits of the evidence. '
                                   'Do not request a tool or repeat a command.')
            elif state.decision_format_pending:
                decision_input += ('\nThe previous decision had an invalid format and was not executed. '
                                   'Return one valid action: an answer or question with text, '
                                   'or a tool action with tool_id and arguments.')
            reviewing = state.answer_review_state == 'pending'
            if reviewing:
                decision_input += ('\nA proposed answer has not been delivered. Review the request and observed '
                                   'evidence: which facts does the request depend on, and could any relevant '
                                   'unexamined authorized source materially change the answer? Read a relevant '
                                   'source if needed, ask if essential information remains unavailable, or '
                                   'give a supported answer. Do not read unrelated sources just to exhaust '
                                   'the catalog.')
            prepared = None
            limits = (0, 4096, 1536, 512) if host.compact_source else (0,)
            for compact_limit in limits:
                try:
                    projection = _project(host, state, compact_limit)
                except ContextBudgetExceeded:
                    if host.compact_source is None:
                        raise
                    continue
                prompt = projection.text + decision_input
                if len(prompt.encode('utf-8')) > host.input_reservation:
                    continue
                try:
                    prepared = (host.prepare_choice(prompt, host.output_reservation)
                                if host.prepare_choice else lambda: host.choose(prompt, host.output_reservation))
                except ValueError as exc:
                    category = _chooser_failure_category(exc)
                    if category == 'input_exceeds_reservation':
                        continue
                    if category == 'cancelled' and _cancelled(host):
                        return _terminal(host, state, 'cancelled', 'cancelled')
                    raise
                break
            if prepared is None:
                return _terminal(host, state, 'blocked', repair_failure or 'candidate_input_reservation_too_small')
            fingerprint = _projection_fingerprint(projection)
            if host.prepare_choice is not None and not _is_fresh(host, state, fingerprint, compact_limit):
                return _terminal(host, state, 'blocked', 'candidate_evidence_changed_or_revoked')
            call_id = 'model-' + uuid4().hex
            amounts = {'actions': 0, 'model_calls': 1, 'input_tokens': host.input_reservation,
                       'output_tokens': host.output_reservation}
            if host.atomic_checkpoint is None:
                budget = host.budgets.load(state.task_id)
                if budget.usage()['in_flight']:
                    return _terminal(host, state, 'blocked', 'candidate_unresolved_reservation')
                host.budgets.reserve(state.task_id, expected_revision=budget.revision, reservation_id=call_id,
                                     kind='model', amounts=amounts, now_ms=host.now_ms())
            else:
                host.atomic_checkpoint.reserve(expected_version=host.atomic_checkpoint.version,
                    reservation_id=call_id, kind='model', amounts=amounts, now_ms=host.now_ms(),
                    projection_fingerprint=fingerprint)
            state.inflight = call_id
            # Consume the single review before dispatch and persist together
            # with the model marker. Crashes cannot silently replay it.
            if reviewing:
                state.answer_review_state = 'used'
            _save(host, state)
        except (BudgetExhausted, EvidenceAccessError, ValueError, KeyError) as exc:
            return _terminal(host, state, 'blocked', repair_failure or type(exc).__name__ + ':' + str(exc))
        if _cancelled(host):
            # Proven undispatched here; retain the reservation conservatively.
            return _terminal(host, state, 'cancelled', 'cancelled')
        if _deadline_elapsed(host):
            return _terminal(host, state, 'blocked', 'candidate_deadline_exceeded')
        if host.atomic_checkpoint is not None and not host.atomic_checkpoint.check_claim(call_id, 'model'):
            return _terminal(host, state, 'blocked', 'candidate_claim_unavailable')
        chooser_failure = ''
        parse_error = None
        try:
            decision = prepared()
        except ModelResponseParseError as exc:
            parse_error = exc
            decision = None
        except Exception as exc:
            chooser_failure = _chooser_failure_category(exc)
            logger.warning('simple_loop_decision_failed category=%s', chooser_failure)
            decision = None
        usage = decision.get('usage', {}) if isinstance(decision, dict) else {}
        if parse_error is not None:
            usage = parse_error.usage
        parse_stage = parse_error.response_stage if parse_error is not None else None
        if not isinstance(usage, dict):
            usage = {}
        try:
            outcome = 'success' if isinstance(decision, dict) else 'failure'
            retry_parse = False
            if host.atomic_checkpoint is None:
                budget = host.budgets.load(state.task_id)
                revision = budget.revision
                budget.settle(call_id, outcome=outcome, input_tokens=usage.get('input_tokens'),
                              output_tokens=usage.get('output_tokens'), parse_stage=parse_stage)
                host.budgets.save(state.task_id, budget, expected_revision=revision)
                if parse_error is not None:
                    retry_parse = state.request_parse_repair(budget)
            else:
                settled = host.atomic_checkpoint.settle_model(expected_version=host.atomic_checkpoint.version,
                    reservation_id=call_id, outcome=outcome, input_tokens=usage.get('input_tokens'),
                    output_tokens=usage.get('output_tokens'), decision=decision, projection_fingerprint=fingerprint,
                    parse_stage=parse_stage)
                budget = settled.budget
                if parse_error is not None:
                    # Settlement and the one retry grant share a transaction.
                    state = settled.state
                    retry_parse = not repairing_parse and state.parse_repair_pending
        except (ValueError, KeyError):
            return _terminal(host, state, 'blocked', 'candidate_model_settlement_unavailable')
        state.inflight = ''
        if host.atomic_checkpoint is None:
            _save(host, state)
        if _cancelled(host):
            return _terminal(host, state, 'cancelled', 'cancelled')
        if parse_error is not None:
            if retry_parse:
                continue
            return _terminal(host, state, 'blocked', 'candidate_model_response_invalid')
        if not isinstance(decision, dict):
            if repairing_parse:
                return _terminal(host, state, 'blocked', repair_failure)
            reasons = {
                'input_exceeds_reservation': 'candidate_input_reservation_too_small',
                'output_exceeds_reservation': 'candidate_output_reservation_too_small',
                'result_invalid': 'candidate_model_response_invalid',
                'provider_model_not_configured': 'candidate_model_not_configured',
                'provider_cli_missing': 'candidate_model_cli_missing',
                'provider_subscription_auth_unverified': 'candidate_model_auth_unverified',
                'provider_authentication_required': 'candidate_model_auth_required',
                'provider_cli_upgrade_required': 'candidate_model_cli_upgrade_required',
                'provider_provider_timeout': 'candidate_model_timeout',
                'provider_empty_response': 'candidate_model_empty_response',
                'provider_invalid_response': 'candidate_model_response_invalid',
            }
            return _terminal(host, state, 'blocked', reasons.get(chooser_failure, 'candidate_model_unavailable'))
        if any((value is not None and (type(value) is not int or value < 0))
               or (value is None and not host.synthetic_allow_unreported_usage)
               for value in (usage.get('input_tokens'), usage.get('output_tokens'))):
            return _terminal(host, state, 'blocked', repair_failure or 'candidate_model_usage_unknown_or_overrun')
        if not budget.within_work_limits():
            return _terminal(host, state, 'blocked', repair_failure or 'candidate_model_usage_overrun')
        if host.atomic_checkpoint is not None and getattr(host.atomic_checkpoint, 'steering_decision_stale', False):
            # Usage was settled, but a user correction arrived during the
            # model call. The atomic checkpoint intentionally kept no selected
            # decision; re-project the current goal before any action.
            if repairing_parse:
                return _terminal(host, state, 'blocked', repair_failure)
            continue
        # A still-authorized source may nevertheless have changed during inference.
        if not _is_fresh(host, state, fingerprint, compact_limit):
            return _terminal(host, state, 'blocked', 'candidate_evidence_changed_or_revoked')
        kind = decision.get('kind')
        if repairing_parse and (not isinstance(kind, str) or kind not in {'answer', 'question'}):
            return _terminal(host, state, 'blocked', repair_failure)
        valid_shape = (isinstance(kind, str) and (kind in {'answer', 'question'} or
                       (kind == 'tool' and isinstance(decision.get('tool_id'), str)
                        and isinstance(decision.get('arguments'), dict))))
        if not valid_shape:
            # Do not infer an answer or execute a malformed action. One generic
            # repair opportunity is budgeted and checkpointed; the bad model
            # output itself is not copied into subsequent prompts or state.
            if state.invalid_decision_retries < 1:
                state.invalid_decision_retries += 1
                state.decision_format_pending = True
                _save(host, state, consume_decision=True)
                continue
            return _terminal(host, state, 'blocked', 'candidate_decision_invalid')
        if state.decision_format_pending:
            state.decision_format_pending = False
            state.parse_repair_pending = False
            _save(host, state)
        if kind in {'answer', 'question'}:
            text = decision.get('text')
            if not isinstance(text, str) or not text.strip():
                return _terminal(host, state, 'blocked', repair_failure or 'candidate_empty_communication')
            if kind == 'question':
                if host.allow_question is not None:
                    try:
                        allowed_question = host.allow_question(text)
                    except Exception:
                        return _terminal(host, state, 'blocked', 'candidate_question_review_unavailable')
                    if allowed_question is not True:
                        return _terminal(host, state, 'blocked', 'candidate_question_not_verified')
                if _deadline_elapsed(host):
                    return _terminal(host, state, 'blocked', 'candidate_deadline_exceeded')
                if not _mark_steering_included(host, decision, fingerprint):
                    return _terminal(host, state, 'blocked', 'candidate_steering_unavailable')
                return _terminal(host, state, 'waiting_user', 'question_delivery_not_integrated', text,
                                 selected_decision=decision, projection_fingerprint=fingerprint)
            if not repairing_parse and host.synthetic_preanswer_review and state.answer_review_state == 'not_started':
                # Hold the proposal; do not persist, return, or pass its text as
                # evidence. The next budgeted decision sees only the current
                # authorized projection and the neutral review request.
                state.answer_review_state = 'pending'
                _save(host, state, consume_decision=True)
                continue
            try:
                # Re-read under the current actor before accepting any response.
                grounded = host.verify_response(state.actor, text, tuple(state.evidence_sources)) is True
            except Exception:
                return _terminal(host, state, 'blocked', 'candidate_verification_unavailable')
            if _cancelled(host):
                return _terminal(host, state, 'cancelled', 'cancelled')
            if not _is_fresh(host, state, fingerprint, compact_limit):
                return _terminal(host, state, 'blocked', 'candidate_evidence_changed_or_revoked')
            if _cancelled(host):
                return _terminal(host, state, 'cancelled', 'cancelled')
            if _deadline_elapsed(host):
                return _terminal(host, state, 'blocked', 'candidate_deadline_exceeded')
            if not grounded:
                if repairing_parse:
                    return _terminal(host, state, 'blocked', repair_failure)
                missing = False
                if host.remaining_advertised_evidence is not None and state.unsupported_retries < 2:
                    try:
                        missing = host.remaining_advertised_evidence(state.actor, tuple(state.evidence_sources)) is True
                    except Exception:
                        missing = False
                elif host.retry_unverified_answer and state.unsupported_retries == 0:
                    missing = True
                if missing:
                    state.unsupported_retries += 1
                    state.recovery_pending = True
                    _save(host, state, consume_decision=True)
                    continue
                return _terminal(host, state, 'blocked', 'candidate_answer_unsupported')
            if not _mark_steering_included(host, decision, fingerprint):
                return _terminal(host, state, 'blocked', 'candidate_steering_unavailable')
            return _terminal(host, state, 'completed', response=text, selected_decision=decision,
                             projection_fingerprint=fingerprint)
        if kind != 'tool' or not isinstance(decision.get('tool_id'), str) or not isinstance(decision.get('arguments'), dict):
            return _terminal(host, state, 'blocked', 'candidate_decision_invalid')
        tool_id, args = decision['tool_id'], deepcopy(decision['arguments'])
        if host.covered_web_fetch is not None and tool_id == 'native.web_fetch':
            try:
                covered = host.covered_web_fetch(state.actor, tool_id, deepcopy(args), tuple(state.evidence_sources))
                if covered is not None and (not isinstance(covered, str) or covered not in state.evidence_sources):
                    raise ValueError('covered_web_fetch_invalid')
            except Exception:
                return _terminal(host, state, 'blocked', 'candidate_retry_policy_unavailable')
            if covered is not None:
                state.reused_web_fetch_ref = covered
                _save(host, state, consume_decision=True)
                continue
        if host.defer_identical_failed_read is not None:
            try:
                defer = host.defer_identical_failed_read(state.actor, tool_id, deepcopy(args), tuple(state.evidence_sources))
                if type(defer) is not bool:
                    raise ValueError('retry_policy_invalid')
            except Exception:
                return _terminal(host, state, 'blocked', 'candidate_retry_policy_unavailable')
            if defer:
                if state.retry_feedback_pending:
                    return _terminal(host, state, 'blocked', 'candidate_retry_deferred')
                state.retry_feedback_pending = True
                _save(host, state, consume_decision=True)
                continue
        if state.retry_feedback_pending:
            state.retry_feedback_pending = False
            _save(host, state)
        registry = getattr(host.context, 'tools', None)
        descriptor = registry.get(tool_id) if registry is not None else None
        from .owner_knowledge_snapshot import OWNER_KNOWLEDGE_TOOL_ID
        if (tool_id == OWNER_KNOWLEDGE_TOOL_ID and
                (host.atomic_checkpoint is None or args != {} or descriptor is None
                 or descriptor.metadata.get('sensitive_content') is not True
                 or not callable(host.read_private_knowledge) or not callable(host.record_private_knowledge))):
            return _terminal(host, state, 'blocked', 'candidate_private_read_unavailable')
        mutation = {
            'native.owner_reminder_create': (host.commit_owner_reminder, 'owner_reminder_unavailable'),
            'native.scheduled_task': (host.commit_owner_schedule, 'owner_schedule_unavailable'),
            OWNER_DISPOSABLE_HANDOFF_TOOL_ID: (host.commit_disposable_handoff, 'owner_disposable_handoff_unavailable'),
        }.get(tool_id)
        if mutation is not None:
            commit, unavailable = mutation
            if (host.atomic_checkpoint is None or not callable(commit)
                    or (tool_id != OWNER_DISPOSABLE_HANDOFF_TOOL_ID
                        and (descriptor is None or descriptor.read_only is not False))
                    or _cancelled(host) or _deadline_elapsed(host)
                    or not _is_fresh(host, state, fingerprint, compact_limit)):
                return _terminal(host, state, 'blocked', unavailable)
            try:
                if not _mark_steering_included(host, decision, fingerprint):
                    return _terminal(host, state, 'blocked', 'candidate_steering_unavailable')
                committed = commit(state, decision, fingerprint)
                if isinstance(committed, CandidateState) and committed.status == 'running':
                    state = committed
                    continue
                return committed
            except MissingPrerequisite as exc:
                if (_cancelled(host) or not _is_fresh(host, state, fingerprint, compact_limit)
                        or exc.code not in FEEDBACK or state.prerequisite_retries >= 2):
                    return _terminal(host, state, 'blocked', 'candidate_prerequisite_unavailable')
                state.prerequisite_feedback = exc.code
                state.prerequisite_retries += 1
                _save(host, state, consume_decision=True)
                continue
            except Exception:
                return _terminal(host, state, 'blocked', unavailable)
        disposable = (tool_id == 'native.bash' and descriptor is not None
                      and descriptor.read_only is False
                      and (host.atomic_checkpoint is None or
                           (isinstance(host.disposable_atomic_session, str) and bool(host.disposable_atomic_session)))
                      and callable(host.authorize_disposable_tool) and callable(host.execute_disposable_tool))
        host_bash = (tool_id == 'native.bash' and not disposable and descriptor is not None
                     and descriptor.read_only is False and host.atomic_checkpoint is not None
                     and callable(host.authorize_host_bash) and callable(host.execute_host_bash))
        if descriptor is None or getattr(getattr(descriptor, 'kind', None), 'value', '') != 'native' or (descriptor.read_only is not True and not disposable and not host_bash):
            return _terminal(host, state, 'blocked', 'candidate_requires_read_only_native_tool')
        def permitted_action():
            if disposable:
                return host.authorize_disposable_tool(state.actor, tool_id, deepcopy(args)) is True
            if host_bash:
                return host.authorize_host_bash(state.actor, host.task.task_id, deepcopy(args)) is True
            return host.authorize_read(state.actor, tool_id, args) is True
        try:
            permitted = permitted_action()
        except Exception:
            permitted = False
        if not permitted:
            return _terminal(host, state, 'blocked', 'candidate_read_denied')
        if _cancelled(host):
            return _terminal(host, state, 'cancelled', 'cancelled')
        if not _mark_steering_included(host, decision, fingerprint):
            return _terminal(host, state, 'blocked', 'candidate_steering_unavailable')
        action_id = 'tool-' + uuid4().hex
        try:
            amounts = {'actions': 1, 'model_calls': 0, 'input_tokens': 0, 'output_tokens': 0}
            if host.atomic_checkpoint is None:
                budget = host.budgets.load(state.task_id)
                host.budgets.reserve(state.task_id, expected_revision=budget.revision, reservation_id=action_id,
                    kind='tool', amounts=amounts, now_ms=host.now_ms())
            else:
                host.atomic_checkpoint.reserve(expected_version=host.atomic_checkpoint.version,
                    reservation_id=action_id, kind='tool', amounts=amounts, now_ms=host.now_ms(),
                    tool_id=tool_id, arguments=args, selected_decision=decision,
                    projection_fingerprint=fingerprint)
            state.inflight = action_id
            _save(host, state)
        except (BudgetExhausted, ValueError, KeyError) as exc:
            return _terminal(host, state, 'blocked', type(exc).__name__ + ':' + str(exc))
        if _cancelled(host):
            return _terminal(host, state, 'cancelled', 'cancelled')
        # Re-check revocation at dispatch; native tool invocation applies its own
        # validation/context but does not substitute for host authorization.
        try:
            permitted = permitted_action()
        except Exception:
            permitted = False
        if not permitted:
            return _terminal(host, state, 'blocked', 'candidate_read_denied')
        if not _is_fresh(host, state, fingerprint, compact_limit):
            return _terminal(host, state, 'blocked', 'candidate_evidence_changed_or_revoked')
        if _cancelled(host):
            return _terminal(host, state, 'cancelled', 'cancelled')
        if _deadline_elapsed(host):
            return _terminal(host, state, 'blocked', 'candidate_deadline_exceeded')
        if host.atomic_checkpoint is not None and not host.atomic_checkpoint.check_claim(
                action_id, 'tool', tool_id=tool_id, arguments=args):
            return _terminal(host, state, 'blocked', 'candidate_claim_unavailable')
        private_digest = None
        from .owner_calendar import OWNER_CALENDAR_TOOL_ID
        from .owner_gmail_search import OWNER_GMAIL_SEARCH_TOOL_ID
        from .owner_outlook import OWNER_OUTLOOK_TOOL_ID
        if tool_id == OWNER_KNOWLEDGE_TOOL_ID:
            from .owner_private_knowledge_read import PrivateKnowledgeRead
            try:
                private = host.read_private_knowledge()
                if (type(private) is not PrivateKnowledgeRead or not isinstance(private.observation, dict)
                        or not isinstance(private.binding_digest, str) or len(private.binding_digest) != 64
                        or any(ch not in '0123456789abcdef' for ch in private.binding_digest)):
                    raise ValueError('private_read_invalid')
                private_digest = private.binding_digest
                result = {'status': 'success', 'result': deepcopy(private.observation)}
            except Exception:
                result = {'status': 'failed', 'result': None}
        elif tool_id in {OWNER_CALENDAR_TOOL_ID, OWNER_GMAIL_SEARCH_TOOL_ID, OWNER_OUTLOOK_TOOL_ID}:
            # Keep provider details outside generic result events, memory and telemetry.
            if host.atomic_checkpoint is None or descriptor.metadata.get('sensitive_content') is not True:
                return _terminal(host, state, 'blocked', 'candidate_private_read_unavailable')
            try:
                payload = ToolInvocationService(context=host.context, task=host.task).execute_or_raise(tool_id, args)
            except Exception:
                if _cancelled(host):
                    return _terminal(host, state, 'cancelled', 'cancelled')
                return _terminal(host, state, 'blocked', 'candidate_private_read_unavailable')
            result = {'status': 'success', 'result': payload}
        elif disposable:
            try:
                result = host.execute_disposable_tool(state.actor, tool_id, deepcopy(args))
                if (not isinstance(result, dict) or result.get('status') not in {'success', 'failed'}
                        or not isinstance(result.get('result'), dict)):
                    raise ValueError('invalid_disposable_result')
            except Exception:
                # An effect may have occurred. Retain the in-flight marker and
                # reservation rather than settling or replaying unknown work.
                return _terminal(host, state, 'blocked', 'candidate_disposable_effect_unconfirmed')
        elif host_bash:
            try:
                result = {'status': 'success', 'result': host.execute_host_bash(state.actor, deepcopy(args), action_id)}
            except Exception:
                # A process may have started. Preserve the durable dispatch
                # marker and reservation for explicit recovery, never replay.
                return _terminal(host, state, 'blocked', 'candidate_host_bash_effect_unconfirmed')
        else:
            result = ToolInvocationService(context=host.context, task=host.task).invoke(tool_id, args, call_id=action_id)
        if host.atomic_checkpoint is None:
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
            permitted = permitted_action()
        except Exception:
            permitted = False
        if not permitted:
            return _terminal(host, state, 'blocked', 'candidate_read_revoked')
        if not _is_fresh(host, state, fingerprint, compact_limit):
            return _terminal(host, state, 'blocked', 'candidate_evidence_changed_or_revoked')
        if result.get('status') != 'success':
            # Explicit, completed read-tool failures are useful observations.
            # Uncaught exceptions, waiting results, and effectful tool failures
            # remain unconfirmed and must not be retried implicitly.
            from .owner_web import web_failure_observation
            failure = web_failure_observation(tool_id, result)
            if failure is None:
                return _terminal(host, state, 'blocked', 'candidate_tool_failed_or_waiting')
            result = {**result, 'result': failure}
        payload = result.get('result')
        # Tool-specific coverage/partial semantics must be verified by the host;
        # this prototype only accepts a deliberately explicit fixture contract.
        try:
            complete = (isinstance(payload, dict) and
                (payload.get('candidate_complete') is True if host.validate_observation is None
                 else host.validate_observation(state.actor, tool_id, deepcopy(args), deepcopy(payload)) is True))
        except Exception:
            complete = False
        if not complete:
            return _terminal(host, state, 'blocked', 'candidate_observation_incomplete')
        # Validation may itself perform I/O. Its success does not preserve the
        # authority, deadline, cancellation or earlier evidence state that was
        # checked before it began. Recheck before recording an observation.
        if _cancelled(host):
            return _terminal(host, state, 'cancelled', 'cancelled')
        if _deadline_elapsed(host):
            return _terminal(host, state, 'blocked', 'candidate_deadline_exceeded')
        try:
            permitted = permitted_action()
        except Exception:
            permitted = False
        if not permitted:
            return _terminal(host, state, 'blocked', 'candidate_read_revoked')
        if not _is_fresh(host, state, fingerprint, compact_limit):
            return _terminal(host, state, 'blocked', 'candidate_evidence_changed_or_revoked')
        try:
            ref = (host.record_private_knowledge(state.actor, tool_id, deepcopy(args), deepcopy(result), private_digest)
                   if private_digest is not None else
                   host.record_evidence(state.actor, result) if host.record_bound_observation is None
                   else host.record_bound_observation(state.actor, tool_id, deepcopy(args), deepcopy(result)))
        except Exception:
            return _terminal(host, state, 'blocked', 'candidate_evidence_recording_unavailable')
        if not isinstance(ref, str) or not ref.strip() or ref == state.goal_source or ref in state.evidence_sources:
            return _terminal(host, state, 'blocked', 'candidate_evidence_reference_invalid')
        if host.atomic_checkpoint is not None:
            extra = ({'knowledge_snapshot_digest': private_digest,
                      'cancellation_checker': host.context.cancellation_checker} if private_digest is not None else {})
            if disposable:
                host.atomic_checkpoint.settle_dojo_bash(expected_version=host.atomic_checkpoint.version,
                    reservation_id=action_id, arguments=args, evidence_ref=ref, observation=payload,
                    session_id=host.disposable_atomic_session, context=host.context)
            elif host_bash:
                host.atomic_checkpoint.settle_host_bash_return(expected_version=host.atomic_checkpoint.version,
                    reservation_id=action_id, arguments=args, evidence_ref=ref,
                    authorize=host.authorize_host_bash, cancellation_checker=host.context.cancellation_checker)
            else:
                host.atomic_checkpoint.settle_read(expected_version=host.atomic_checkpoint.version,
                    reservation_id=action_id, tool_id=tool_id, arguments=args, evidence_ref=ref, observation=payload, **extra)
        state.evidence_sources.append(ref)
        if (state.prerequisite_feedback == 'owner_clock' and tool_id == 'native.owner_clock'
                or state.prerequisite_feedback == 'owner_schedules' and tool_id == 'native.owner_schedules'):
            state.prerequisite_feedback = ''
        # Reading one source does not establish that the prior answer was
        # supported. Preserve the generic feedback until a terminal outcome.
        state.inflight = ''
        _save(host, state)
    return state
