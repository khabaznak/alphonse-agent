"""Development-only prescription adapter for the unregistered single loop.

The injected chooser is a local scripted double. It sees the goal and observed
source contents only, never the corpus oracle. This is not a live benchmark or
an authorization, response-grounding, delivery, or clinical implementation.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .simplified_loop import AdapterUnavailable, Run

_CASES = frozenset({
    'development-prescription-morning-day-three',
    'development-prescription-morning-final-day',
    'development-prescription-morning-after-course',
})
_PATHS = frozenset({'household/index.md', 'household/prescription.md',
                    'household/treatment-start.md'})
_ACTOR = 'synthetic-evaluation-user'


class PrescriptionCandidateAdapter:
    """Execute the actual candidate with isolated read-only fixture callbacks.

    `chooser` takes projected evidence and returns a candidate decision. It must
    be a local deterministic test double, and must not read the fixture root or
    independent oracle. It is not a model provider and no usage is billed.
    """
    offline = True
    revision = 'working-tree:single-loop:prescription-development-v1'

    def __init__(self, chooser):
        self.chooser = chooser

    def __call__(self, case, fixture_root, emit):
        if case.get('id') not in _CASES:
            raise AdapterUnavailable('prescription_development_case_not_supported')
        files = case.get('fixture', {}).get('files')
        if not isinstance(files, dict) or set(files) != _PATHS or 'oracle' in case:
            raise AdapterUnavailable('prescription_development_fixture_invalid')

        from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
        from alphonse.agent_v2.core.intelligence.task_state import TaskState
        from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateHost, CandidateState, run_candidate
        from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits, SQLiteTaskBudgetStore, TaskBudget
        from alphonse.agent_v2.core.messages import InMemoryMessageQueue
        from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition

        root = Path(fixture_root).resolve()
        sources = {'goal': case['goal']}
        registry = InMemoryToolRegistry()
        descriptor = ToolDescriptor('native.fixture_read', 'fixture_read', ToolKind.NATIVE,
            argument_schema={'type': 'object', 'properties': {'path': {'type': 'string'}},
                             'required': ['path'], 'additionalProperties': False}, read_only=True)

        def permitted(actor, tool, args):
            return (actor == _ACTOR and tool == 'native.fixture_read'
                    and set(args) == {'path'} and args.get('path') in _PATHS)

        def read(args):
            emit({'kind': 'tool', 'tool': 'native.fixture_read', 'arguments': dict(args)})
            if not permitted(_ACTOR, 'native.fixture_read', args):
                raise PermissionError('fixture_read_denied')
            path = root / args['path']
            resolved = path.resolve(strict=True)
            if root not in resolved.parents or path.is_symlink() or not resolved.is_file():
                raise PermissionError('fixture_read_outside_root')
            emit({'kind': 'source_read', 'source': args['path']})
            return {'candidate_complete': True, 'path': args['path'], 'content': resolved.read_text()}

        registry.register(ToolDefinition(descriptor, read))
        task = TaskState(task_id='prescription-development', user=_ACTOR)
        context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry)

        def choose(text, cap):
            # The chooser must derive its action from projection alone. These
            # byte counts are conservative synthetic units, not provider usage.
            value = self.chooser(text)
            if not isinstance(value, dict):
                return value
            decision = dict(value)
            output_units = len(json.dumps(decision, ensure_ascii=False).encode())
            if output_units > cap:
                raise ValueError('fixture_output_exceeds_cap')
            units = {'input_tokens': len(text.encode()), 'output_tokens': output_units}
            emit({'kind': 'system_two', 'estimated_usage': units})
            decision['usage'] = units
            return decision

        def record(actor, result):
            payload = result['result']
            ref = 'read:' + payload['path']
            sources[ref] = json.dumps(payload, ensure_ascii=False)
            return ref

        def verify(actor, text, refs):
            # This synthetic gate requires the source files to have been read;
            # it does not determine correctness. The independent harness oracle
            # grades the final structured response after the run.
            return (actor == _ACTOR and set(refs) == {'read:' + p for p in _PATHS}
                    and isinstance(text, str) and bool(text.strip()))

        def units(actions=0, calls=0, inp=0, out=0):
            return {'actions': actions, 'model_calls': calls, 'input_tokens': inp, 'output_tokens': out}

        with TemporaryDirectory(prefix='prescription-candidate-') as directory:
            store = SQLiteTaskBudgetStore(str(Path(directory) / 'budget.db'))
            store.create(task.task_id, TaskBudget(BudgetLimits(units(5, 8, 40000, 4096),
                         units(0, 1, 200, 100), 100000, 1000), started_at_ms=0))
            checkpoint_path = Path(directory) / 'checkpoint.json'
            host = CandidateHost(task, context, store,
                lambda snapshot: checkpoint_path.write_text(json.dumps(snapshot)), choose,
                lambda actor, ref: actor == _ACTOR and ref in sources,
                lambda actor, ref: sources[ref], permitted, record, verify, lambda: 100,
                input_reservation=4096, output_reservation=512, projection_budget=4096)
            state = run_candidate(CandidateState(task.task_id, _ACTOR, 'goal'), host)
        return Run(status=state.status, response=state.response, reason=state.blocker or None,
                   measured_kinds=('system_two', 'jev', 'tool', 'question', 'source_read'))
