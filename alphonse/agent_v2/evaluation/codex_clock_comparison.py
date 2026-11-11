"""Opt-in, synthetic clock comparison using Alphonse's Codex provider.

This test-level switch does not alter the daemon or production routing. Each
arm runs the real processor against a new local clock and the same supplied
provider configuration. Model calls are not made at import. Caller must verify
Codex subscription authentication before dispatch; the harness is not a billing
or network sandbox and does not establish production authorization or delivery.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any, Callable

from .loopback_clock_pilot import LoopbackClockPilot, evaluate
from .clock_responses_attempt import _reason_category
from .prescription_http_processors import LocalSource
from .token_usage_diagnostics import usage_diagnostics

ACTOR = 'synthetic-evaluation-user'
GOAL = ('Read the synthetic clock and provide the local time and date in its '
        'timezone. Answer as a JSON object with local_time, local_date and timezone.')


def _summarize_events(events: list[dict]) -> dict:
    """Keep measured and estimated usage distinct; partial totals are unknown."""
    calls = [event for event in events if event.get('kind') == 'system_two']
    def total(direction, field):
        if not calls:
            return None
        values = [(event.get('usage_diagnostics') or {}).get(direction, {}).get(field)
                  for event in calls]
        return sum(values) if all(type(value) is int and value >= 0 for value in values) else None
    return {'system_two_calls': len(calls),
            'tool_calls': sum(event.get('kind') == 'tool' for event in events),
            'provider_input_tokens': total('input', 'provider_tokens'),
            'provider_output_tokens': total('output', 'provider_tokens'),
            'input_tokens_measured_or_estimated': total('input', 'tokens'),
            'output_tokens_measured_or_estimated': total('output', 'tokens'),
            'estimate_note': 'Per-call provenance is in events; mixed totals include estimates and may omit provider framing or reasoning.'}


class _TrackedProvider:
    """Record observed V3 calls without changing provider behavior or storing text."""
    def __init__(self, provider, emit, trace=None):
        self.provider, self.emit, self.trace = provider, emit, trace
        self.supports_image_input = getattr(provider, 'supports_image_input', False)

    def _call(self, name, request):
        from alphonse.agent_v2.core.inference.openai_codex import (
            _json_envelope, _markdown_envelope, _tool_planning_envelope,
        )
        observed = {'generate_json': _json_envelope, 'generate_markdown': _markdown_envelope,
                    'plan_tool_call': _tool_planning_envelope}[name](request)
        start = monotonic()
        try:
            result = getattr(self.provider, name)(request)
        except Exception:
            self.emit({'kind': 'system_two', 'status': 'failed',
                       'elapsed_seconds': monotonic() - start,
                       'usage_diagnostics': usage_diagnostics(input_text=observed, output_text=None)})
            raise
        if self.trace is not None:
            value = getattr(result, 'json_value', None)
            tool = value.get('tool_id') if isinstance(value, dict) else None
            self.trace({'event': 'model_decision', 'arm': 'v3',
                        'purpose': getattr(getattr(request, 'purpose', None), 'value', 'unknown'),
                        'tool': tool if isinstance(tool, str) and tool in {'native.fixture_read', 'native.respond', 'native.ask_question'} else None,
                        'structured_output': isinstance(value, dict)})
        self.emit({'kind': 'system_two', 'status': 'returned',
                   'elapsed_seconds': monotonic() - start,
                   'usage_diagnostics': usage_diagnostics(input_text=observed,
                       output_text=getattr(result, 'raw_response', None) or getattr(result, 'content', None) or None,
                       provider_usage=getattr(result, 'usage', None))})
        return result

    def generate_json(self, request):
        return self._call('generate_json', request)

    def generate_markdown(self, request):
        return self._call('generate_markdown', request)

    def plan_tool_call(self, request):
        return self._call('plan_tool_call', request)


def _run_v3(*, url: str, provider: Any, jev: Any, model: str,
            emit: Callable[[dict], None], cancelled: Callable[[], bool], trace=None,
            goal: str = GOAL, source_description: str = 'Read the synthetic clock source.',
            allowed_source_names: tuple[str, ...] = ('index',),
            required_source_names: tuple[str, ...] = ('index',)):
    """Run the actual V3 processor; injected provider and Jev are not mocked here."""
    if jev is None:
        raise ValueError('v3_requires_system_one')
    from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
    from alphonse.agent_v2.core.inference import InferenceRouter, ModelProfile
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import ToolDefinition
    from alphonse.agent_v2.core.tools.registry.native.respond import build_respond_tool_definition
    from .deterministic_adapters import _Action, _RunState, _Scenario, _registry
    from .simplified_loop import Run

    if (not isinstance(required_source_names, tuple) or not required_source_names
            or not set(required_source_names) <= set(allowed_source_names)):
        raise ValueError('invalid_required_sources')
    source = LocalSource(url, allowed_source_names=allowed_source_names)
    def diagnostic(event):
        if trace is not None:
            try:
                trace(event)
            except Exception:
                pass
    with TemporaryDirectory(prefix='codex-clock-v3-') as directory:
        action = _Action('native.fixture_read', 'project_record_search', {}, role='read_clock', behavior='read')
        acceptance = ('Answer from the synthetic clock.' if goal == GOAL
                      else 'Answer from the authorized synthetic source.')
        state = _RunState(_Scenario((action,), acceptance, ''), Path(directory), 'hierarchical_v3')
        registry = _registry(state)
        source_schema = ({'const': 'index'} if allowed_source_names == ('index',)
                         else {'type': 'string', 'enum': list(allowed_source_names)})
        descriptor = ToolDescriptor('native.fixture_read', 'fixture_read', ToolKind.NATIVE,
            description=source_description,
            argument_schema={'type': 'object', 'properties': {'source': source_schema},
                             'required': ['source'], 'additionalProperties': False},
            metadata={'v3_capabilities': ['project_record_search']}, read_only=True)

        def read(args):
            name = args.get('source') if isinstance(args, dict) else None
            name = name if isinstance(name, str) and name in allowed_source_names else None
            diagnostic({'event': 'source_read_attempt', 'arm': 'v3', 'source': name})
            if cancelled():
                raise ValueError('clock_cancelled_before_read')
            emit({'kind': 'tool', 'tool': 'native.fixture_read'})
            result = source.read(args)
            if cancelled():
                raise ValueError('clock_cancelled_after_read')
            emit({'kind': 'source_read', 'source': args['source']})
            diagnostic({'event': 'source_read_observed', 'arm': 'v3', 'source': name})
            return result

        registry.register(ToolDefinition(descriptor, read, descriptor.argument_schema,
                                         enabled=True, accepts_context=False))
        # Use the product's real response descriptor, schema and local payload
        # builder. The deterministic fixture's generic read-only descriptor
        # misrepresents a user response; this tool only prepares a payload and
        # does not send a message or establish user-visible delivery.
        response_tool = build_respond_tool_definition()
        def respond(args):
            emit({'kind': 'tool', 'tool': 'native.respond'})
            result = response_tool.callable(args)
            diagnostic({'event': 'response_prepared', 'arm': 'v3'})
            return result
        registry.register(ToolDefinition(response_tool.descriptor, respond, response_tool.argument_schema,
                                         enabled=response_tool.enabled, accepts_context=response_tool.accepts_context))
        router = InferenceRouter(provider=_TrackedProvider(provider, emit, trace), default_profile=ModelProfile(
            provider='openai_codex', model=model, profile_id='codex-clock'))
        task = TaskState(task_id='codex-v3-clock', message_id='codex-v3-clock', user=ACTOR,
                         project_id='synthetic-clock', goal=goal,
                         intelligence_engine='hierarchical_v3', intelligence_schema_version=3)
        context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry,
                                  inference=router, system_one=jev, cancellation_checker=cancelled)
        result = HierarchicalCAPDProcessor().process(task, context)
        final = TaskState.from_dict(result.snapshot.metadata.get('task_state') or task.to_dict())
        response = final.metadata.get('prepared_user_response') or {}
        reason = str((final.outcome or {}).get('reason') or '') or None
        diagnostic({'event': 'terminal', 'arm': 'v3',
                    'status': final.status if final.status in {'completed', 'failed', 'cancelled', 'waiting_user', 'blocked', 'running'} else 'other',
                    'response_accepted': bool(response), 'reason_category': _reason_category(reason)})
        return Run(status=final.status, response=str(response.get('message') or ''),
                   reason=reason)


def _run_simple(*, url: str, provider: Any, model: str,
                emit: Callable[[dict], None], cancelled: Callable[[], bool], trace=None,
                goal: str = GOAL, source_description: str = 'Read the synthetic clock source.',
                allowed_source_names: tuple[str, ...] = ('index',),
                required_source_names: tuple[str, ...] | None = ('index',),
                verification_mode: str = 'fixture_diagnostic'):
    """Run the actual read-only candidate with a subscription-backed chooser."""
    from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateHost, CandidateState, run_candidate
    from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits, SyntheticEvaluationBudgetStore, TaskBudget
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
    from .codex_candidate_chooser import CodexSubscriptionChooser
    from .simplified_loop import Run

    if verification_mode not in {'fixture_diagnostic', 'matched_provenance'}:
        raise ValueError('invalid_verification_mode')
    if verification_mode == 'fixture_diagnostic':
        if (not isinstance(required_source_names, tuple) or not required_source_names
                or not set(required_source_names) <= set(allowed_source_names)):
            raise ValueError('invalid_required_sources')
    elif required_source_names is not None:
        # A matched run must never pass the oracle's coverage key into the loop.
        raise ValueError('matched_mode_disallows_required_sources')
    source = LocalSource(url, allowed_source_names=allowed_source_names)
    evidence = {'goal': goal}
    registry = InMemoryToolRegistry()
    source_schema = ({'const': 'index'} if allowed_source_names == ('index',)
                     else {'type': 'string', 'enum': list(allowed_source_names)})
    descriptor = ToolDescriptor('native.fixture_read', 'fixture_read', ToolKind.NATIVE,
        description=source_description,
        argument_schema={'type': 'object', 'properties': {'source': source_schema},
                         'required': ['source'], 'additionalProperties': False}, read_only=True)

    def permit(actor, tool, args):
        return (actor == ACTOR and tool == 'native.fixture_read' and isinstance(args, dict)
                and set(args) == {'source'} and isinstance(args['source'], str)
                and args['source'] in allowed_source_names)

    def read(args):
        if not permit(ACTOR, 'native.fixture_read', args) or cancelled():
            raise PermissionError('fixture_read_denied')
        emit({'kind': 'tool', 'tool': 'native.fixture_read'})
        result = source.read(args)
        emit({'kind': 'source_read', 'source': args['source']})
        return result

    registry.register(ToolDefinition(descriptor, read))
    task = TaskState(task_id='codex-simple-clock', user=ACTOR)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, cancellation_checker=cancelled)
    def record(actor, result):
        payload = result['result']
        ref = 'read:' + payload['source']
        evidence[ref] = json.dumps(payload, ensure_ascii=False)
        return ref
    def remaining_advertised(actor, refs):
        if actor != ACTOR:
            return False
        unread = [name for name in allowed_source_names if 'read:' + name not in refs]
        # Evaluate every authorization before choosing to recover; exceptions
        # propagate to the loop's fail-closed policy boundary.
        permissions = [permit(actor, 'native.fixture_read', {'source': name}) is True for name in unread]
        return any(permissions)

    def units(actions=0, calls=0, inp=0, out=0):
        return {'actions': actions, 'model_calls': calls, 'input_tokens': inp, 'output_tokens': out}

    chooser = CodexSubscriptionChooser(provider=provider, model=model, max_input_tokens=65536,
                                        emit=emit, cancellation_checker=cancelled,
                                        allowed_source_names=allowed_source_names,
                                        source_description=source_description)
    class TracedChooser:
        max_input_tokens = chooser.max_input_tokens
        def __call__(self, projection, output_cap):
            decision = chooser(projection, output_cap)
            if trace is not None:
                kind = decision.get('kind')
                tool = decision.get('tool_id')
                text = decision.get('text') if isinstance(decision.get('text'), str) else ''
                lowered = text.casefold()
                trace({'event': 'model_decision', 'arm': 'simple',
                       'kind': kind if isinstance(kind, str) and kind in {'tool', 'answer', 'question'} else 'other',
                       'tool': tool if tool == 'native.fixture_read' else None,
                       'mentions_tool_availability': kind == 'question' and
                           any(term in lowered for term in ('enable', 'unavailable', 'cannot access', 'not available'))})
            return decision
    with TemporaryDirectory(prefix='codex-clock-candidate-') as directory:
        budgets = SyntheticEvaluationBudgetStore(str(Path(directory) / 'budget.db'))
        now = lambda: int(monotonic() * 1000)
        budgets.create(task.task_id, TaskBudget(BudgetLimits(units(4, 11, 655560, 163940),
                       units(0, 1, 200, 100), 100000, 1000), started_at_ms=now()))
        def verify(actor, text, refs):
            if not (actor == ACTOR and bool(text.strip()) and bool(refs)
                    and all(ref in evidence for ref in refs)):
                return False
            if verification_mode == 'matched_provenance':
                return True
            return {'read:' + name for name in required_source_names} <= set(refs)

        host = CandidateHost(task, context, budgets, lambda _snapshot: None, TracedChooser(),
            lambda actor, ref: actor == ACTOR and ref in evidence,
            lambda actor, ref: evidence[ref], permit, record,
            verify,
            now, input_reservation=65536, output_reservation=16384, projection_budget=65536,
            deadline_checker=cancelled, synthetic_allow_unreported_usage=True,
            remaining_advertised_evidence=(remaining_advertised if verification_mode == 'fixture_diagnostic' else None))
        state = run_candidate(CandidateState(task.task_id, ACTOR, 'goal'), host)
    return Run(status=state.status, response=state.response, reason=state.blocker or None)


def run_clock_arm(*, arm: str, provider: Any, model: str, jev: Any = None,
                  instant_utc: str = '2026-04-12T14:00:00+00:00',
                  timezone: str = 'America/Mexico_City', timeout_seconds: float = 100,
                  private_trace: Callable[[dict], None] | None = None) -> dict:
    """Run one selected processor on a fresh synthetic fixture; caller controls authorization.

    The same provider object/model must be supplied to both separately executed
    arms. No direct OpenAI API transport is selected here. This function does not
    verify subscription login, sandbox arbitrary injected providers, establish a
    hard timeout, or send real user-facing messages.
    """
    from .codex_subscription_provider import CodexSubscriptionProvider
    if arm not in {'v3', 'simple'}:
        raise ValueError('unsupported_clock_arm')
    if not isinstance(model, str) or not model.strip():
        raise ValueError('model_required')
    if private_trace is not None and not callable(private_trace):
        raise ValueError('invalid_trace_callback')
    if type(provider) is not CodexSubscriptionProvider or provider.model != model:
        raise ValueError('matching_codex_subscription_provider_required')
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds < float('inf'):
        raise ValueError('invalid_timeout')
    start = monotonic()
    deadline = start + timeout_seconds
    events: list[dict] = []
    def cancelled():
        return monotonic() >= deadline
    def safe_trace(event):
        if private_trace is not None:
            try:
                private_trace(event)
            except Exception:
                pass  # Diagnostic failures must not alter the processor.
    with LoopbackClockPilot(instant_utc, timezone) as fixture:
        try:
            run = (_run_v3(url=fixture.url, provider=provider, jev=jev, model=model, emit=events.append, cancelled=cancelled, trace=safe_trace if private_trace else None)
                   if arm == 'v3' else
                   _run_simple(url=fixture.url, provider=provider, model=model, emit=events.append, cancelled=cancelled, trace=safe_trace if private_trace else None))
            error = None
        except Exception as exc:
            # No provider or source exception text can leak secrets into summary.
            run = None
            error = type(exc).__name__
    try:
        answer = json.loads(run.response) if run else None
    except (TypeError, ValueError):
        answer = None
    oracle = evaluate(fixture, response=answer, expected_instant_utc=instant_utc, expected_timezone=timezone)
    return {'arm': arm, 'model': model, 'status': run.status if run else 'error',
            'reason_category': _reason_category(run.reason) if run else None,
            'passed': bool(run and run.status == 'completed' and oracle['passed']),
            'error_type': error, 'elapsed_seconds': monotonic() - start,
            'oracle': oracle, 'events': events, 'metrics': _summarize_events(events), 'delivery': 'unassessed',
            'qualification': 'unassessed',
            'limitations': ['Synthetic fixture and tools, not the full desktop conversation path.',
                            'Subscription login checks do not prove billing mode during dispatch.',
                            'Candidate locally checks a 16384-byte visible output bound; V3 is not identically bounded.',
                            'Missing candidate provider usage is unknown; synthetic accounting retains the full reservation.',
                            'Jev is separate and required by the V3 arm.',
                            'Timeout is cooperative; in-flight work may overrun.']}


def compare_clock(*, provider: Any, model: str, jev_factory: Callable[[], Any],
                  repetitions: int = 1, timeout_seconds: float = 100,
                  private_trace: Callable[[dict], None] | None = None) -> dict:
    """Run both arms on fresh clocks; never a production rollout or implicit provider call.

    This function calls the explicitly supplied provider and is not invoked at
    import. The user must authorize live Codex and Jev use before calling it with
    enabled providers. Two repetitions alternate arm order to reduce order bias.
    The caller must retain output as synthetic evaluation data only.
    """
    from .codex_subscription_provider import CodexSubscriptionProvider
    if type(provider) is not CodexSubscriptionProvider or provider.model != model:
        raise ValueError('matching_codex_subscription_provider_required')
    if not callable(jev_factory) or type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise ValueError('invalid_comparison_configuration')
    results = []
    for repetition in range(1, repetitions + 1):
        order = ('v3', 'simple') if repetition % 2 else ('simple', 'v3')
        for arm in order:
            try:
                jev = jev_factory() if arm == 'v3' else None
                result = run_clock_arm(arm=arm, provider=provider, model=model, jev=jev,
                                       timeout_seconds=timeout_seconds, private_trace=private_trace)
            except Exception as exc:
                # Keep an unavailable run in the denominator without printing secrets.
                result = {'arm': arm, 'model': model, 'status': 'error', 'passed': None,
                          'error_type': type(exc).__name__, 'elapsed_seconds': None,
                          'oracle': None, 'events': [], 'delivery': 'unassessed',
                          'qualification': 'unassessed'}
            results.append({'repetition': repetition, **result})
    return {'kind': 'synthetic_clock_v3_vs_simple', 'model': model, 'runs': results,
            'qualification': 'unassessed',
            'limitations': ['Fresh fixtures; this does not run the full Alphonse desktop flow.',
                            'Verify ChatGPT subscription authentication before any live run.',
                            'Codex usage may be estimated, not provider reported.',
                            'Jev and actual user delivery remain separate observability boundaries.']}
