"""Experimental Responses-backed candidate path for the *synthetic* clock.

This module does not establish authorization, isolation, provider pricing, a
non-bypassable spending cap or production safety. It does not launch by itself.
The caller must obtain explicit approval for provider/model, synthetic data,
retention, run count and spend before enabling any network sender. In particular,
the ledger and caller-supplied input counter are not independent cost controls.

The candidate is the actual unregistered read-only prototype. Its only exposed
fixture capability is the loopback clock's ``index`` source. The evaluator must
independently grade the response; reaching ``completed`` is not a verified pass.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Callable

from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest, InferenceResult
from .responses_pilot_transport import PilotProviderResultError
from .token_usage_diagnostics import usage_diagnostics

_ACTOR = 'synthetic-evaluation-user'
_GOAL = ('Read the synthetic clock and provide the local time and date in its '
         'timezone. Answer as a JSON object with local_time, local_date and timezone.')
_INSTRUCTIONS = (
    'Return one JSON decision. Read the synthetic clock with '
    '{"kind":"tool","tool_id":"native.fixture_read","arguments":{"source":"index"}}. '
    'Return the final JSON object as a string in {"kind":"answer","text":"..."}. '
    'Do not claim clock evidence that is absent.\n\n'
)


class ClockResponsesChooser:
    """Bridge to a bounded Responses transport without pretending it is offline.

    ``enabled`` defaults to false. It is a local switch, not proof of user
    approval, a network sandbox or an enforceable global spending boundary.
    The transport must report actual usage; it is not replaced by estimates.
    """
    def __init__(self, transport, *, max_input_tokens: int, emit: Callable, enabled: bool = False,
                 cancellation_checker: Callable | None = None):
        if type(max_input_tokens) is not int or max_input_tokens <= 0 or not callable(emit):
            raise ValueError('invalid_clock_chooser_configuration')
        if (type(enabled) is not bool or not callable(getattr(transport, 'generate_json', None))
                or (cancellation_checker is not None and not callable(cancellation_checker))):
            raise ValueError('invalid_clock_chooser_configuration')
        self.transport, self.max_input_tokens, self.emit, self.enabled = transport, max_input_tokens, emit, enabled
        self.cancellation_checker = cancellation_checker

    def __call__(self, projection: str, max_output_tokens: int):
        if not self.enabled:
            raise ValueError('clock_model_dispatch_disabled')
        if self.cancellation_checker is not None and self.cancellation_checker():
            raise ValueError('clock_model_cancelled_before_dispatch')
        if not isinstance(projection, str) or type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError('invalid_clock_model_request')
        prompt = _INSTRUCTIONS + projection
        if len(prompt.encode('utf-8')) > self.max_input_tokens:
            raise ValueError('clock_model_input_cap_exceeded')
        request = InferenceRequest(prompt=prompt, purpose=InferencePurpose.TACTICAL_ACTION,
                                   metadata={'evaluation': 'synthetic_clock'},
                                   cancel_checker=self.cancellation_checker)
        prompt_sha256 = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        started = monotonic()
        transport_failed = False
        result = None
        try:
            result = self.transport.generate_json(request, max_input_tokens=self.max_input_tokens,
                                                  max_output_tokens=max_output_tokens)
        except Exception as exc:
            transport_failed = True
            reported = dict(exc.provider_usage) if type(exc) is PilotProviderResultError else None
            self.emit({'kind': 'system_two', 'status': 'failed', 'provider_usage': reported,
                       'prompt_sha256': prompt_sha256,
                       'duration_ms': round((monotonic() - started) * 1000, 3),
                       'usage_diagnostics': usage_diagnostics(input_text=prompt, output_text=None,
                                                              provider_usage=reported)})
        if transport_failed:
            # A transport exception can retain authorization headers or source
            # content. Raise outside its handler, without exception chaining.
            raise ValueError('clock_model_transport_failed')
        if not isinstance(result, InferenceResult) or not isinstance(result.json_value, dict):
            raw = result.usage if isinstance(result, InferenceResult) and isinstance(result.usage, dict) else {}
            invalid_usage = {key: raw.get(key) if type(raw.get(key)) is int and raw[key] >= 0 else None
                             for key in ('input_tokens', 'output_tokens')}
            self.emit({'kind': 'system_two', 'status': 'invalid', 'provider_usage': invalid_usage,
                       'prompt_sha256': prompt_sha256,
                       'duration_ms': round((monotonic() - started) * 1000, 3),
                       'usage_diagnostics': usage_diagnostics(input_text=prompt, output_text=None,
                                                              provider_usage=invalid_usage)})
            raise ValueError('invalid_clock_model_decision')
        raw = result.usage if isinstance(result.usage, dict) else {}
        usage = {key: raw.get(key) if type(raw.get(key)) is int and raw[key] >= 0 else None
                 for key in ('input_tokens', 'output_tokens')}
        decision = dict(result.json_value)
        decision.pop('usage', None)
        decision['usage'] = usage
        cancelled_after = self.cancellation_checker is not None and self.cancellation_checker()
        self.emit({'kind': 'system_two', 'status': 'cancelled_after_dispatch' if cancelled_after else 'returned',
                   'prompt_sha256': prompt_sha256,
                   'provider_usage': dict(usage),
                   'duration_ms': round((monotonic() - started) * 1000, 3),
                   'usage_diagnostics': usage_diagnostics(input_text=prompt,
                       output_text=result.raw_response if isinstance(result.raw_response, str) else result.content or None,
                       provider_usage=usage)})
        if cancelled_after:
            raise ValueError('clock_model_cancelled_after_dispatch')
        return decision


def run_clock_candidate(*, base_url: str, chooser: ClockResponsesChooser, emit: Callable,
                        cancellation_checker: Callable | None = None):
    """Execute one synthetic clock attempt with exactly one authorized source.

    This is not a live-ready launcher: the chooser's switch and transport are
    supplied by the caller and cannot establish experiment approval or isolation.
    No wall-clock timeout, response delivery or production integration is claimed.
    """
    if (not isinstance(chooser, ClockResponsesChooser) or not callable(emit)
            or chooser.cancellation_checker is not cancellation_checker):
        raise ValueError('invalid_clock_runner_configuration')
    from .prescription_http_processors import LocalSource
    from .simplified_loop import Run
    from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateHost, CandidateState, run_candidate
    from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits, SQLiteTaskBudgetStore, TaskBudget
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition

    source = LocalSource(base_url, allowed_source_names=('index',))
    evidence = {'goal': _GOAL}
    registry = InMemoryToolRegistry()
    descriptor = ToolDescriptor('native.fixture_read', 'fixture_read', ToolKind.NATIVE,
        argument_schema={'type': 'object', 'properties': {'source': {'type': 'string'}},
                         'required': ['source'], 'additionalProperties': False}, read_only=True)

    def permitted(actor, tool, args):
        return actor == _ACTOR and tool == 'native.fixture_read' and args == {'source': 'index'}

    def read(args):
        if not permitted(_ACTOR, 'native.fixture_read', args):
            raise PermissionError('fixture_source_denied')
        emit({'kind': 'tool', 'tool': 'native.fixture_read', 'source': 'index'})
        value = source.read(args)
        emit({'kind': 'source_read', 'source': 'index'})
        return value

    registry.register(ToolDefinition(descriptor, read))
    task = TaskState(task_id='synthetic-clock-responses', user=_ACTOR)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry,
                              cancellation_checker=cancellation_checker)

    def record(actor, result):
        evidence['read:index'] = json.dumps(result['result'], ensure_ascii=False)
        return 'read:index'

    def units(actions=0, calls=0, inp=0, out=0):
        return {'actions': actions, 'model_calls': calls, 'input_tokens': inp, 'output_tokens': out}

    with TemporaryDirectory(prefix='candidate-clock-responses-') as directory:
        budgets = SQLiteTaskBudgetStore(str(Path(directory) / 'budget.db'))
        # A fixed synthetic clock would disable elapsed-time budget checks. The
        # task duration uses monotonic time; the fixture's wall clock remains
        # separately frozen. This does not interrupt an already-running call.
        def budget_now_ms():
            return int(monotonic() * 1000)
        started_at_ms = budget_now_ms()
        deadline_ms = started_at_ms + 100000
        # Ten ordinary calls plus the existing independent final reserve. The
        # provider ledger remains the outer cap for actual model dispatches.
        budgets.create(task.task_id, TaskBudget(BudgetLimits(units(4, 11, 655560, 5220),
                       units(0, 1, 200, 100), 100000, 1000), started_at_ms=started_at_ms))

        class TimedChooser:
            # Preserve the real input cap checked by the candidate host.
            max_input_tokens = chooser.max_input_tokens

            def __call__(self, projection, output_cap):
                if budget_now_ms() >= deadline_ms:
                    raise ValueError('clock_task_deadline_exceeded')
                decision = chooser(projection, output_cap)
                if budget_now_ms() >= deadline_ms:
                    # Do not accept late tool calls or answers. This check
                    # cannot interrupt a currently running provider request.
                    raise ValueError('clock_task_deadline_exceeded')
                return decision

        host = CandidateHost(task, context, budgets, lambda _snapshot: None, TimedChooser(),
            lambda actor, ref: actor == _ACTOR and ref in evidence,
            lambda actor, ref: evidence[ref], permitted, record,
            lambda actor, text, refs: actor == _ACTOR and bool(text.strip()) and refs == ('read:index',),
            budget_now_ms, input_reservation=65536, output_reservation=512, projection_budget=65536,
            deadline_checker=lambda: budget_now_ms() >= deadline_ms)
        state = run_candidate(CandidateState(task.task_id, _ACTOR, 'goal'), host)
    return Run(status=state.status, response=state.response, reason=state.blocker or None)
