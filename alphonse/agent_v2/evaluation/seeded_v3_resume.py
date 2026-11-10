"""One V3 seeded-checkpoint replay, using only disposable synthetic effects.

The seeded checkpoint simulates a prior successful lookup; it does not prove
an actual process crash or production persistence. The V3 engine and fixture
mutation execute for real. This adapter is not applicable to arbitrary cases.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .simplified_loop import AdapterUnavailable, Run


class V3SeededResumeAdapter:
    offline = True
    revision = 'working-tree:hierarchical-v3:seeded-resume-fixture-v1'

    def __call__(self, case, fixture_root, emit):
        if case.get('id') != 'v3-seeded-resume':
            raise AdapterUnavailable('seeded_resume_case_not_supported')
        from alphonse.agent_v2.core.core import CoreLoopContext
        from alphonse.agent_v2.core.inference import InferencePurpose, InferenceResult, InferenceRouter, ModelProfile
        from alphonse.agent_v2.core.intelligence.task_state import TaskState
        from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor
        from alphonse.agent_v2.core.messages import InMemoryMessageQueue
        from alphonse.agent_v2.core.tools.registry import ToolDefinition
        from .deterministic_adapters import _DeterministicProvider, _ReplayJev, _RunState, _registry, _restore_checkpoint, _scenario

        project = fixture_root / 'project'
        project.mkdir()
        original = fixture_root / 'records'
        original.rename(project / 'records')
        scenario = _scenario('restart-mid-phase')
        state = _RunState(scenario=scenario, root=project, engine='hierarchical_v3')
        state.v3_index = 1  # The checkpoint below includes the completed first lookup.

        class Provider(_DeterministicProvider):
            def generate_json(self, request):
                if request.purpose == InferencePurpose.TASK_UNDERSTANDING:
                    return InferenceResult(json_value={'requested_outcome': case['goal'], 'supplied_parameters': [],
                        'constraints': [], 'unresolved_dependencies': [], 'superseded_constraints': []})
                return super().generate_json(request)

        def telemetry(event):
            if event.get('event_type') == 'inference':
                emit({'kind': 'system_two', 'estimated_usage': {
                    'input_tokens': event.get('input_tokens_estimate'),
                    'output_tokens': event.get('output_tokens_estimate')}})

        registry = _registry(state)
        for descriptor in registry.list():
            definition = registry.get_definition(descriptor.tool_id)
            def invoke(args, original=definition.callable, tool_id=descriptor.tool_id):
                emit({'kind': 'tool', 'tool': tool_id, 'arguments': dict(args)})
                if tool_id == 'fixture.read':
                    emit({'kind': 'source_read', 'source': args.get('path')})
                if tool_id == 'native.ask_question':
                    emit({'kind': 'question'})
                return original(args)
            registry.register(ToolDefinition(definition.descriptor, invoke, definition.argument_schema,
                                             enabled=definition.enabled, accepts_context=False))
        context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry,
            inference=InferenceRouter(provider=Provider(state), default_profile=ModelProfile(
                provider='deterministic', model='replay-v1', profile_id='offline-replay'), telemetry_sink=telemetry),
            # Do not claim Jev counts: this fixture has no complete event wrapper.
            system_one=_ReplayJev({a.tool_id for a in scenario.actions if not a.read_only}), telemetry_sink=telemetry)
        task = TaskState(task_id='v3-resume-fixture', message_id='fixture-message', user='fixture-user',
                         project_id='fixture-project', goal=case['goal'], intelligence_engine='hierarchical_v3',
                         intelligence_schema_version=3)
        task.set_acceptance_contract_from_markdown(f'- [ ] {scenario.acceptance}')
        _restore_checkpoint(task, scenario, 'hierarchical_v3')
        try:
            with TemporaryDirectory(prefix='alphonse-v3-seeded-state-') as temporary:
                checkpoint = Path(temporary) / 'task.json'
                checkpoint.write_text(json.dumps(task.to_dict()), encoding='utf-8')
                resumed = TaskState.from_dict(json.loads(checkpoint.read_text(encoding='utf-8')))
                emit({'kind': 'restart', 'seeded': True})
                outcome = HierarchicalCAPDProcessor().process(resumed, context)
                final = TaskState.from_dict(outcome.snapshot.metadata.get('task_state') or resumed.to_dict())
                prepared = final.metadata.get('prepared_user_response') or {}
                return Run(status=final.status, response=str(prepared.get('message') or ''),
                           reason=str((final.outcome or {}).get('reason') or '') or None,
                           measured_kinds=('system_two', 'tool', 'question', 'source_read', 'restart'))
        finally:
            (project / 'records').rename(original)
            project.rmdir()
