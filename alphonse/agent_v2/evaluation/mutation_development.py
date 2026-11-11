"""Offline seeded-resume mutation comparison and candidate safety probe.

This development replay uses a disposable file and scripted decisions.  The
candidate's read-only restriction is deliberately preserved: it is not an
available mutation arm.  Neither the seeded checkpoint nor the synthetic tool
authorizes production effects or simulates an actual crash.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from .simplified_loop import _run, load_frozen

_ROOT = Path(__file__).resolve().parents[3]
_CORPUS = _ROOT / 'tests/fixtures/single_loop_comparison/v3_seeded_resume_v1.json'
_FLAG = 'ALPHONSE_V3_CONTEXT_PROJECTIONS'


def _candidate_rejection_probe():
    """Actually request a non-read-only tool and independently read back state."""
    from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateHost, CandidateState, run_candidate
    from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits, SQLiteTaskBudgetStore, TaskBudget
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


def _worker():
    from .seeded_v3_resume import V3SeededResumeAdapter
    from alphonse.agent_v2.core.intelligence.v3.projections import enabled
    case = load_frozen(_CORPUS)[0]['cases'][0]
    result = _evaluate_v3_attempts(_run(case, V3SeededResumeAdapter()))
    return {'projection_enabled': enabled(), 'result': result}


def _evaluate_v3_attempts(result):
    """Require a single observed edit invocation as well as the file oracle.

    The fixture tool event is emitted before invocation, so this checks attempts;
    final read-back checks state, not the number of externally applied effects.
    """
    result = dict(result)
    events = result.get('events')
    count = (sum(event.get('kind') == 'tool' and event.get('tool') == 'native.exact_text_edit'
                 for event in events if isinstance(event, dict)) if isinstance(events, list) else None)
    result['mutation_attempts'] = count
    result['passed'] = result.get('passed') is True and count == 1
    return result


def _v3_arm(projection_enabled):
    env = {key: value for key, value in os.environ.items()
           if key in {'PATH', 'LANG', 'LC_ALL', 'TMPDIR', 'SYSTEMROOT'}}
    env[_FLAG] = '1' if projection_enabled else '0'
    try:
        child = subprocess.run([sys.executable, '-m', __name__, '--worker'], cwd=str(_ROOT),
            env=env, capture_output=True, text=True, timeout=30, check=False)
        if child.returncode:
            raise RuntimeError('worker_failed')
        payload = json.loads(child.stdout)
        if payload['projection_enabled'] is not projection_enabled:
            raise ValueError('projection_flag_mismatch')
        return payload['result']
    except Exception as exc:
        return {'availability': 'error', 'passed': None, 'reason': type(exc).__name__,
                'events': None, 'oracle': None}


def compare_mutation():
    """Run both V3 variants and expose the candidate's missing capability."""
    return {'experiment_kind': 'offline_scripted_seeded_mutation_development',
        'arms': {'current_v3': _v3_arm(False), 'targeted_v3_projection': _v3_arm(True),
                 'single_loop': {'availability': 'unavailable', 'passed': None,
                                 'reason': 'non_read_only_effects_not_implemented',
                                 'safety_probe': _candidate_rejection_probe()}},
        'decision_qualified': False, 'delivery': 'unassessed',
        'limitations': ['Checkpoint is seeded; this does not simulate a crash.',
                        'Provider and Jev are scripted; no autonomous behavior is measured.',
                        'Effects and authorization are synthetic, not production enforcement.',
                        'Exactly one edit attempt and final file state are observed, not an independent applied-effect count.',
                        'Candidate cannot execute mutations and its safety probe is not a task success.']}


if __name__ == '__main__' and sys.argv[1:] == ['--worker']:
    print(json.dumps(_worker(), ensure_ascii=False))
