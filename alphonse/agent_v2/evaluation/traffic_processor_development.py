"""Scripted processor integration for the public synthetic traffic portal.

All arms receive the same preselected actions and answer. This is a control-flow
fixture, not a browser, autonomous model benchmark, delivery check or production
authorization boundary. Fixture answers are public; no provider is called.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from .decision_traffic_development import DEVELOPMENT_CASES, SyntheticTrafficPortal, evaluate_traffic

_WORKSPACE = Path(__file__).resolve().parents[3]
_FLAG = 'ALPHONSE_V3_CONTEXT_PROJECTIONS'
_ARMS = ('current_v3', 'targeted_v3_projection', 'single_loop')
_ACTOR = 'synthetic-evaluation-user'


def _perform(portal, args):
    if args == {'operation': 'inspect'}:
        return portal.inspect()
    if set(args) == {'operation', 'field', 'subject'} and args['operation'] == 'search':
        return portal.search(field=args['field'], subject=args['subject'])
    raise ValueError('invalid_synthetic_portal_action')


def _v3(case, portal, actions, answer, emit):
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.inference import InferencePurpose, InferenceResult, InferenceRouter, ModelProfile
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import ToolDefinition
    from .deterministic_adapters import _Action, _Scenario, _RunState, _registry, _ReplayJev, _DeterministicProvider
    scenario = _Scenario(actions=tuple(_Action('fixture.portal', 'project_record_search', args,
                        role='portal_' + str(i)) for i, args in enumerate(actions)),
                        acceptance='Inspect the synthetic portal and report the requested status.',
                        final_response=json.dumps(answer))
    with TemporaryDirectory(prefix='traffic-v3-') as directory:
        state = _RunState(scenario=scenario, root=Path(directory), engine='hierarchical_v3')
        class Provider(_DeterministicProvider):
            def generate_json(self, request):
                if request.purpose == InferencePurpose.TASK_UNDERSTANDING:
                    return InferenceResult(json_value={'requested_outcome': 'Check synthetic tickets for ' + case.subject,
                        'supplied_parameters': [], 'constraints': [], 'unresolved_dependencies': [], 'superseded_constraints': []})
                return super().generate_json(request)
        class Judge(_ReplayJev):
            def evaluate(self, **kw):
                emit({'kind': 'jev'}); return super().evaluate(**kw)
            def curate_request_tools(self, **kw):
                emit({'kind': 'jev'}); return super().curate_request_tools(**kw)
            def select_plan_tools(self, **kw):
                emit({'kind': 'jev'}); return super().select_plan_tools(**kw)
            def triage_plan_messages(self, **kw):
                emit({'kind': 'jev'}); return super().triage_plan_messages(**kw)
            def judge_act(self, **kw):
                emit({'kind': 'jev'}); return super().judge_act(**kw)
            def evaluate_tactical_progress(self, **kw):
                emit({'kind': 'jev'}); return super().evaluate_tactical_progress(**kw)
        def telemetry(event):
            if event.get('event_type') == 'inference': emit({'kind': 'system_two'})
        registry = _registry(state)
        definition = registry.get_definition('fixture.portal')
        original = definition.callable
        def invoke(args):
            emit({'kind': 'tool', 'tool': 'fixture.portal', 'arguments': dict(args)})
            original(args)  # The replay provider tracks completed fixture calls.
            return {'candidate_complete': True, 'observation': _perform(portal, args)}
        registry.register(ToolDefinition(definition.descriptor, invoke, definition.argument_schema,
                                         enabled=definition.enabled, accepts_context=False))
        router = InferenceRouter(provider=Provider(state), default_profile=ModelProfile(
            provider='deterministic', model='replay-v1', profile_id='offline-replay'), telemetry_sink=telemetry)
        context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, inference=router,
                                  system_one=Judge(), telemetry_sink=telemetry)
        task = TaskState(task_id='traffic-v3', message_id='traffic-message', user=_ACTOR,
                         project_id='fixture-project', goal='Check synthetic tickets for ' + case.subject,
                         intelligence_engine='hierarchical_v3', intelligence_schema_version=3)
        result = HierarchicalCAPDProcessor().process(task, context)
        final = TaskState.from_dict(result.snapshot.metadata.get('task_state') or task.to_dict())
        response = final.metadata.get('prepared_user_response') or {}
        return final.status, str(response.get('message') or '')


def _candidate(case, portal, actions, answer, emit):
    from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateHost, CandidateState, run_candidate
    from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits, SQLiteTaskBudgetStore, TaskBudget
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
    sources = {'goal': 'Check synthetic tickets for ' + case.subject}
    registry = InMemoryToolRegistry()
    descriptor = ToolDescriptor('fixture.portal', 'fixture.portal', ToolKind.NATIVE,
        argument_schema={'type': 'object', 'additionalProperties': True}, read_only=True)
    def permit(actor, tool, args):
        return actor == _ACTOR and tool == 'fixture.portal' and isinstance(args, dict)
    def invoke(args):
        emit({'kind': 'tool', 'tool': 'fixture.portal', 'arguments': dict(args)})
        return {'candidate_complete': True, 'observation': _perform(portal, args)}
    registry.register(ToolDefinition(descriptor, invoke))
    task = TaskState(task_id='traffic-candidate', user=_ACTOR)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry)
    decisions = iter([*({'kind': 'tool', 'tool_id': 'fixture.portal', 'arguments': args} for args in actions),
                      {'kind': 'answer', 'text': json.dumps(answer)}])
    def choose(text, cap):
        value = next(decisions)
        units = len(json.dumps(value).encode())
        if units > cap: raise ValueError('fixture_output_exceeds_cap')
        emit({'kind': 'system_two'})
        return {**value, 'usage': {'input_tokens': len(text.encode()), 'output_tokens': units}}
    def record(actor, result):
        ref = 'observation:' + str(len(sources))
        sources[ref] = json.dumps(result['result'])
        return ref
    def units(actions=0, calls=0, inp=0, out=0):
        return {'actions': actions, 'model_calls': calls, 'input_tokens': inp, 'output_tokens': out}
    with TemporaryDirectory(prefix='traffic-candidate-') as directory:
        store = SQLiteTaskBudgetStore(str(Path(directory) / 'budget.db'))
        store.create(task.task_id, TaskBudget(BudgetLimits(units(8, 10, 50000, 5000),
                     units(0, 1, 200, 100), 100000, 1000), started_at_ms=0))
        host = CandidateHost(task, context, store, lambda snapshot: None, choose,
            lambda actor, ref: actor == _ACTOR and ref in sources, lambda actor, ref: sources[ref],
            permit, record, lambda actor, text, refs: actor == _ACTOR and bool(refs) and bool(text),
            lambda: 100, input_reservation=4096, output_reservation=512, projection_budget=4096)
        final = run_candidate(CandidateState(task.task_id, _ACTOR, 'goal'), host)
    return final.status, final.response


def _worker(request):
    from alphonse.agent_v2.core.intelligence.v3.projections import enabled
    case = next((x for x in DEVELOPMENT_CASES if x.name == request['case']), None)
    if case is None: return {'availability': 'unavailable', 'passed': None, 'reason': 'unknown_case'}
    # Session refresh changes portal state. These prototype arms expose only a
    # read-only fixture tool; do not smuggle refresh through that authority.
    if any(action.get('operation') == 'refresh_session' for action in request['actions']):
        return {'availability': 'unavailable', 'passed': None, 'reason': 'session_refresh_requires_stateful_tool'}
    portal = SyntheticTrafficPortal(case)
    events = []
    try:
        status, text = (_candidate(case, portal, request['actions'], request['answer'], events.append)
                        if request['arm'] == 'single_loop' else
                        _v3(case, portal, request['actions'], request['answer'], events.append))
        try: response = json.loads(text)
        except (TypeError, ValueError): response = None
        oracle = evaluate_traffic(case, response, portal)
        return {'availability': 'available', 'processor_status': status,
                'passed': status == 'completed' and oracle['passed'], 'portal_oracle': oracle,
                'response': response, 'events': events, 'portal_audit': portal.audit(),
                'projection_enabled': enabled(), 'delivery': 'unassessed'}
    except Exception as exc:
        return {'availability': 'error', 'passed': None, 'reason': type(exc).__name__,
                'events': events, 'portal_audit': portal.audit(), 'delivery': 'unassessed'}


def run_traffic_arm(case_name, arm, *, actions, answer):
    """Execute one isolated scripted arm; intended for per-attempt journaling."""
    if arm not in _ARMS:
        raise ValueError('unknown_arm')
    if not isinstance(actions, list) or not actions or not isinstance(answer, dict):
        raise ValueError('scripted_actions_and_answer_required')
    env = {k: v for k, v in os.environ.items() if k in {'PATH', 'LANG', 'LC_ALL', 'TMPDIR', 'SYSTEMROOT'}}
    env[_FLAG] = '1' if arm == 'targeted_v3_projection' else '0'
    request = {'case': case_name, 'arm': arm, 'actions': actions, 'answer': answer}
    try:
        child = subprocess.run([sys.executable, '-m', __name__, '--worker'], cwd=str(_WORKSPACE),
            env=env, input=json.dumps(request), capture_output=True, text=True, timeout=30, check=False)
        if child.returncode: raise RuntimeError('worker_failed')
        result = json.loads(child.stdout)
        if result.get('availability') == 'available' and result.get('projection_enabled') is not (arm == 'targeted_v3_projection'):
            raise ValueError('projection_flag_mismatch')
        return result
    except Exception as exc:
        return {'availability': 'error', 'passed': None, 'reason': type(exc).__name__,
                'delivery': 'unassessed', 'events': None, 'portal_audit': None}


def compare_traffic(case_name, *, actions, answer):
    """Execute three isolated scripted processor arms, preserving unavailable/error outcomes."""
    results = {arm: run_traffic_arm(case_name, arm, actions=actions, answer=answer) for arm in _ARMS}
    return {'experiment_kind': 'offline_scripted_traffic_processor_development', 'case': case_name,
            'arms': results, 'decision_qualified': False,
            'limitations': ['Preselected scripts and answers, no autonomous reasoning or provider usage.',
                            'Synthetic in-process portal; no browser or production authorization.',
                            'Session refresh is unsupported by these read-only processor arms.',
                            'Delivery, time and resource usage are not qualified.']}


if __name__ == '__main__' and sys.argv[1:] == ['--worker']:
    print(json.dumps(_worker(json.load(sys.stdin))))
