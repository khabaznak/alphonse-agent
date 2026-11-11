"""Scripted V3 prescription-development adapter, isolated by process.

Executes the actual V3 processor with the same synthetic sources as the candidate.
Scripted actions, answer, and Jev judgments do not measure agent intelligence,
real model usage, production authorization, response delivery, or clinical safety.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

from .simplified_loop import AdapterUnavailable, Run

_CASES = frozenset({'development-prescription-morning-day-three',
                    'development-prescription-morning-final-day',
                    'development-prescription-morning-after-course'})
_PATHS = ('household/index.md', 'household/prescription.md', 'household/treatment-start.md')
_FLAG = 'ALPHONSE_V3_CONTEXT_PROJECTIONS'
_WORKSPACE = Path(__file__).resolve().parents[3]


class PrescriptionV3Adapter:
    """V3 read-only development replay with explicitly supplied scripted answer.

    The caller supplies a mapping from development case IDs to scripted answers;
    neither worker nor provider receives the independent oracle. Projection
    variants run in fresh interpreters; declared offline is not a network sandbox.
    """
    offline = True

    def __init__(self, *, projection_enabled: bool, scripted_answers: dict[str, str]):
        if type(projection_enabled) is not bool:
            raise ValueError('projection_enabled_must_be_boolean')
        if not isinstance(scripted_answers, dict) or any(
                key not in _CASES or not isinstance(value, str)
                for key, value in scripted_answers.items()):
            raise ValueError('invalid_scripted_answers')
        self.projection_enabled = projection_enabled
        self.scripted_answers = dict(scripted_answers)
        self.revision = 'working-tree:v3:prescription-development:projection-' + ('on' if projection_enabled else 'off')

    def __call__(self, case, fixture_root, emit):
        if case.get('id') not in _CASES or case['id'] not in self.scripted_answers:
            raise AdapterUnavailable('prescription_script_not_configured')
        if 'oracle' in case or set(case.get('fixture', {}).get('files', {})) != set(_PATHS):
            raise AdapterUnavailable('prescription_development_fixture_invalid')
        env = {k: v for k, v in os.environ.items() if k in {'PATH', 'LANG', 'LC_ALL', 'TMPDIR', 'SYSTEMROOT'}}
        env[_FLAG] = '1' if self.projection_enabled else '0'
        payload = {'case': case, 'root': str(fixture_root), 'answer': self.scripted_answers[case['id']]}
        child = subprocess.run([sys.executable, '-m', __name__, '--worker'], cwd=str(_WORKSPACE),
                               env=env, input=json.dumps(payload), capture_output=True, text=True,
                               timeout=30, check=False)
        if child.returncode != 0:
            raise RuntimeError('prescription_v3_worker_failed')
        try:
            result = json.loads(child.stdout)
            if result['projection_enabled'] is not self.projection_enabled:
                raise ValueError('prescription_v3_flag_mismatch')
            for event in result['events']:
                emit(event)
            return Run(**result['run'])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError('prescription_v3_worker_invalid') from exc


def _execute(case, root, answer, emit):
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.inference import InferencePurpose, InferenceResult, InferenceRouter, ModelProfile
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import ToolDefinition
    from .deterministic_adapters import _Action, _Scenario, _RunState, _registry, _ReplayJev, _DeterministicProvider

    actions = tuple(_Action('fixture.read', 'project_record_search', {'path': path},
                            role=f'read_{i}', behavior='read') for i, path in enumerate(_PATHS))
    scenario = _Scenario(actions=actions, acceptance='Read the synthetic sources and provide the requested schedule.',
                         final_response=answer)
    state = _RunState(scenario=scenario, root=Path(root), engine='hierarchical_v3')

    class Provider(_DeterministicProvider):
        def generate_json(self, request):
            if request.purpose == InferencePurpose.TASK_UNDERSTANDING:
                return InferenceResult(json_value={'requested_outcome': case['goal'], 'supplied_parameters': [],
                    'constraints': [], 'unresolved_dependencies': [], 'superseded_constraints': []})
            return super().generate_json(request)

    class Judge(_ReplayJev):
        def evaluate(self, **kw):
            emit({'kind': 'jev', 'purpose': 'check'})
            return super().evaluate(**kw)
        def curate_request_tools(self, **kw):
            emit({'kind': 'jev', 'purpose': 'relevance'})
            return super().curate_request_tools(**kw)
        def select_plan_tools(self, **kw):
            emit({'kind': 'jev', 'purpose': 'legacy_relevance'})
            return super().select_plan_tools(**kw)
        def triage_plan_messages(self, **kw):
            emit({'kind': 'jev', 'purpose': 'triage'})
            return super().triage_plan_messages(**kw)
        def judge_act(self, **kw):
            emit({'kind': 'jev', 'purpose': 'act'})
            return super().judge_act(**kw)
        def evaluate_tactical_progress(self, **kw):
            emit({'kind': 'jev', 'purpose': 'progress'})
            return super().evaluate_tactical_progress(**kw)

    def telemetry(event):
        if event.get('event_type') == 'inference':
            emit({'kind': 'system_two', 'estimated_usage': {
                'input_tokens': event.get('input_tokens_estimate'),
                'output_tokens': event.get('output_tokens_estimate')}})

    registry = _registry(state)
    for desc in registry.list():
        definition = registry.get_definition(desc.tool_id)
        def invoke(args, original=definition.callable, tool_id=desc.tool_id):
            emit({'kind': 'tool', 'tool': tool_id, 'arguments': dict(args)})
            if tool_id == 'fixture.read':
                if set(args) != {'path'} or args.get('path') not in _PATHS:
                    raise PermissionError('fixture_read_denied')
                result = original(args)
                emit({'kind': 'source_read', 'source': args['path']})
                return result
            return original(args)
        registry.register(ToolDefinition(definition.descriptor, invoke, definition.argument_schema,
                                         enabled=definition.enabled, accepts_context=False))
    router = InferenceRouter(provider=Provider(state), default_profile=ModelProfile(
        provider='deterministic', model='replay-v1', profile_id='offline-replay'), telemetry_sink=telemetry)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, inference=router,
                              system_one=Judge(), telemetry_sink=telemetry)
    task = TaskState(task_id='prescription-v3', message_id='prescription-message', user='synthetic-evaluation-user',
                     project_id='fixture-project', goal=case['goal'], intelligence_engine='hierarchical_v3',
                     intelligence_schema_version=3)
    result = HierarchicalCAPDProcessor().process(task, context)
    final = TaskState.from_dict(result.snapshot.metadata.get('task_state') or task.to_dict())
    response = final.metadata.get('prepared_user_response') or {}
    return Run(status=final.status, response=str(response.get('message') or ''),
               reason=str((final.outcome or {}).get('reason') or '') or None,
               measured_kinds=('system_two', 'jev', 'tool', 'question', 'source_read'))


def _worker():
    from alphonse.agent_v2.core.intelligence.v3.projections import enabled
    request = json.load(sys.stdin)
    events = []
    try:
        run = _execute(request['case'], request['root'], request['answer'], events.append)
    except Exception as exc:
        run = Run(status='error', error=type(exc).__name__)
    print(json.dumps({'run': asdict(run), 'events': events, 'projection_enabled': enabled()}))


if __name__ == '__main__' and sys.argv[1:] == ['--worker']:
    _worker()
