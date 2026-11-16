"""Offline simple-loop mutation safety probe.

This development test uses a disposable file and scripted decisions. The
candidate's read-only restriction is deliberately preserved: it is not an
available mutation arm. The synthetic tool does not authorize production effects.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory




def _candidate_rejection_probe():
    """Actually request a non-read-only tool and independently read back state."""
    from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateHost, CandidateState, run_candidate
    from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import BudgetLimits, SQLiteTaskBudgetStore, TaskBudget
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition

    def units(actions=0, calls=0, inp=0, out=0):
        return {'actions': actions, 'model_calls': calls, 'input_tokens': inp, 'output_tokens': out}

    with TemporaryDirectory(prefix='mutation-rejection-') as temporary:
        root = Path(temporary)
        record = root / 'item.md'
        record.write_text('Status: open\n', encoding='utf-8')
        before = record.read_bytes()
        invoked = []
        registry = InMemoryToolRegistry()
        descriptor = ToolDescriptor('native.fixture_edit', 'fixture_edit', ToolKind.NATIVE,
            argument_schema={'type': 'object', 'properties': {}, 'additionalProperties': False}, read_only=False)
        def edit(_args):
            invoked.append(True)
            record.write_text('Status: complete\n', encoding='utf-8')
            return {'candidate_complete': True}
        registry.register(ToolDefinition(descriptor, edit))
        task = TaskState(task_id='mutation-rejection', user='synthetic-user')
        context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry)
        store = SQLiteTaskBudgetStore(str(root / 'budget.db'))
        store.create(task.task_id, TaskBudget(BudgetLimits(units(2, 2, 10000, 1000),
                     units(0, 1, 200, 100), 100000, 1000), started_at_ms=0))
        def choose(text, _cap):
            return {'kind': 'tool', 'tool_id': 'native.fixture_edit', 'arguments': {},
                    'usage': {'input_tokens': len(text.encode()), 'output_tokens': 50}}
        host = CandidateHost(task, context, store, lambda _: None, choose,
            lambda actor, ref: actor == 'synthetic-user' and ref == 'goal',
            lambda actor, ref: 'Update the synthetic record.',
            lambda *_: True, lambda *_: 'unused', lambda *_: False, lambda: 100,
            input_reservation=1024, output_reservation=128)
        state = run_candidate(CandidateState(task.task_id, task.user, 'goal'), host)
        after = record.read_bytes()
        passed = (state.status == 'blocked' and state.blocker == 'candidate_requires_read_only_native_tool'
                  and not invoked and before == after)
        return {'passed': passed, 'processor_status': state.status, 'reason': state.blocker,
                'edit_calls': len(invoked), 'independent_file_unchanged': before == after}


def compare_mutation():
    """Report the simple-loop mutation boundary using disposable state."""
    return {'experiment_kind': 'offline_scripted_mutation_development',
        'arms': {'single_loop': {'availability': 'unavailable', 'passed': None,
                                 'reason': 'non_read_only_effects_not_implemented',
                                 'safety_probe': _candidate_rejection_probe()}},
        'decision_qualified': False, 'delivery': 'unassessed',
        'limitations': ['Synthetic mutation rejection is not a task success or production effect verification.']}
