"""Offline scripted V3 replay for synthetic follow-through development cases.

Actual V3 runs in a credential-free child with context projections explicitly
off or on. Source actions and response are supplied scripts, not model decisions.
An optional post-processor bridge simulates receipt; it is not product delivery.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from .decision_followthrough_development import CASES, Case

_NAMES = {case.name for case in CASES}
_WORKSPACE = Path(__file__).resolve().parents[3]
_FLAG = 'ALPHONSE_V3_CONTEXT_PROJECTIONS'


def _terminal_matches(status: str, response: str) -> bool:
    """Do not credit a completed question or a result still waiting for input."""
    try:
        message = json.loads(response)
    except (TypeError, ValueError):
        return False
    if not isinstance(message, dict):
        return False
    return ((message.get('kind') == 'result' and status == 'completed') or
            (message.get('kind') == 'handoff' and status == 'waiting_user'))


def run_v3_fixture(case: Case, *, projection_enabled: bool, scripted_sources: list[str],
                   scripted_response: str, simulate_delivery: bool = False) -> dict:
    """Run one disposable fixture; scripts must not be supplied by the oracle.

    The caller specifies source actions and output before execution. The bridge
    does not establish user delivery, authorization, autonomy or real latency.
    """
    if not isinstance(case, Case) or case.name not in _NAMES:
        raise ValueError('unsupported_development_case')
    if type(projection_enabled) is not bool or type(simulate_delivery) is not bool:
        raise ValueError('invalid_boolean_option')
    if (not isinstance(scripted_sources, list) or
            any(not isinstance(source, str) for source in scripted_sources) or
            not isinstance(scripted_response, str)):
        raise ValueError('invalid_script')
    # Empty-action phases are unsupported by the replay planner. Cancellation
    # is represented as a pending but never-executed source read.
    if not scripted_sources:
        return {'availability': 'unavailable', 'reason': 'empty_replay_not_supported',
                'decision_qualified': False}
    env = {k: v for k, v in os.environ.items()
           if k in {'PATH', 'LANG', 'LC_ALL', 'TMPDIR', 'SYSTEMROOT'}}
    env[_FLAG] = '1' if projection_enabled else '0'
    payload = {'name': case.name, 'goal': case.goal, 'sources': scripted_sources,
               'response': scripted_response, 'bridge': simulate_delivery}
    proc = subprocess.run([sys.executable, '-m', __name__, '--worker'],
        cwd=str(_WORKSPACE), env=env, input=json.dumps(payload), capture_output=True,
        text=True, timeout=30, check=False)
    if proc.returncode:
        raise RuntimeError('followthrough_v3_worker_failed')
    try:
        result = json.loads(proc.stdout)
        if result['projection_enabled'] is not projection_enabled:
            raise ValueError('followthrough_v3_flag_mismatch')
        return result
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError('followthrough_v3_worker_invalid') from exc


def _execute(payload: dict) -> dict:
    from tempfile import TemporaryDirectory
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.inference import InferencePurpose, InferenceResult, InferenceRouter, ModelProfile
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import ToolDefinition
    from .decision_followthrough_development import FollowThroughFixture, evaluate
    from .deterministic_adapters import _Action, _Scenario, _RunState, _registry, _ReplayJev, _DeterministicProvider
    from alphonse.agent_v2.core.intelligence.v3.projections import enabled

    case = Case(payload['name'], payload['goal'])
    fixture = FollowThroughFixture(case)
    events = []
    actions = tuple(_Action('fixture.read', 'project_record_search', {'path': source},
                            role=f'read_{i}', behavior='read')
                    for i, source in enumerate(payload['sources']))
    scenario = _Scenario(actions=actions, acceptance='Inspect authorized synthetic sources and respond.',
                         final_response=payload['response'])
    with TemporaryDirectory(prefix='v3-followthrough-') as directory:
        state = _RunState(scenario=scenario, root=Path(directory), engine='hierarchical_v3')

        class Provider(_DeterministicProvider):
            def generate_json(self, request):
                if request.purpose == InferencePurpose.TASK_UNDERSTANDING:
                    return InferenceResult(json_value={'requested_outcome': case.goal, 'supplied_parameters': [],
                        'constraints': [], 'unresolved_dependencies': [], 'superseded_constraints': []})
                return super().generate_json(request)

        class Judge(_ReplayJev):
            def evaluate(self, **kw):
                events.append({'kind': 'jev'}); return super().evaluate(**kw)
            def curate_request_tools(self, **kw):
                events.append({'kind': 'jev'}); return super().curate_request_tools(**kw)
            def select_plan_tools(self, **kw):
                events.append({'kind': 'jev'}); return super().select_plan_tools(**kw)
            def triage_plan_messages(self, **kw):
                events.append({'kind': 'jev'}); return super().triage_plan_messages(**kw)
            def judge_act(self, **kw):
                events.append({'kind': 'jev'}); return super().judge_act(**kw)
            def evaluate_tactical_progress(self, **kw):
                events.append({'kind': 'jev'}); return super().evaluate_tactical_progress(**kw)

        def telemetry(event):
            if event.get('event_type') == 'inference':
                events.append({'kind': 'system_two'})

        registry = _registry(state)
        original = registry.get_definition('fixture.read')
        def lookup(arguments):
            events.append({'kind': 'tool', 'tool': 'fixture.read', 'arguments': dict(arguments)})
            state.tool_calls.append('fixture.read')
            source = arguments.get('path')
            if set(arguments) != {'path'} or source not in {'primary', 'backup'}:
                fixture.record_denied_read(source if isinstance(source, str) else '<invalid>')
                raise PermissionError('fixture_lookup_denied')
            return fixture.read(source)
        registry.register(ToolDefinition(original.descriptor, lookup, original.argument_schema,
                                         enabled=original.enabled, accepts_context=False))
        router = InferenceRouter(provider=Provider(state), default_profile=ModelProfile(
            provider='deterministic', model='replay-v1', profile_id='offline-replay'), telemetry_sink=telemetry)
        context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, inference=router,
            system_one=Judge(), telemetry_sink=telemetry,
            cancellation_checker=lambda: case.name == 'cancelled')
        task = TaskState(task_id='followthrough-v3', message_id='fixture-message', user='synthetic-user',
            project_id='fixture-project', goal=case.goal, intelligence_engine='hierarchical_v3',
            intelligence_schema_version=3)
        result = HierarchicalCAPDProcessor().process(task, context)
        final = TaskState.from_dict(result.snapshot.metadata.get('task_state') or task.to_dict())
    response = str((final.metadata.get('prepared_user_response') or {}).get('message') or '')
    reason = str((final.outcome or {}).get('reason') or '') or None
    # The fixture contract uses a canonical cancellation reason. Preserve the
    # processor's exact reason separately and normalize only its known literal.
    oracle_reason = 'cancelled' if final.status == 'cancelled' and reason == 'Task cancelled.' else reason
    bridge = 'not_requested'
    if payload['bridge'] and final.status in {'completed', 'waiting_user'}:
        try:
            message = json.loads(response)
        except (ValueError, TypeError):
            bridge = 'invalid_structured_response'
        else:
            bridge = fixture.send('actor_private', message)
    outcome = evaluate(case, fixture, processor_status=final.status, processor_reason=oracle_reason)
    terminal_ok = (final.status == 'cancelled' and oracle_reason == 'cancelled') if case.name == 'cancelled' else _terminal_matches(final.status, response)
    outcome['terminal_semantics'] = terminal_ok
    outcome['passed'] = outcome['passed'] and terminal_ok
    return {'availability': 'available', 'projection_enabled': enabled(),
        'processor_status': final.status, 'processor_reason': reason,
        'processor_response': response, 'events': events, 'fixture_audit': fixture.audit(),
        'fixture_outcome': outcome,
        'simulated_bridge': bridge, 'product_delivery': 'unassessed',
        'decision_qualified': False, 'real_time_deadline': 'unassessed',
        'limitations': ['Scripted provider and Jev; no autonomous model or usage.',
                        'Bridge delivery is simulated outside the processor.',
                        'Synthetic fixture clock is static; real latency is unmeasured.',
                        'No production authorization or notification guarantee.']}


def _worker():
    from alphonse.agent_v2.core.intelligence.v3.projections import enabled
    payload = json.load(sys.stdin)
    try:
        result = _execute(payload)
    except Exception as exc:
        result = {'availability': 'error', 'reason': type(exc).__name__,
                  'projection_enabled': enabled(), 'decision_qualified': False,
                  'product_delivery': 'unassessed', 'real_time_deadline': 'unassessed',
                  'events': None, 'fixture_audit': None,
                  'observation_status': 'unknown_after_worker_error'}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__' and sys.argv[1:] == ['--worker']:
    _worker()
