"""Paired, scripted read-only fixture adapters for the unregistered prototype.

Both adapters execute real processors, not real model providers. These fixture
callbacks are deliberately narrow and are NOT suitable for household traffic.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from .simplified_loop import AdapterUnavailable, Run

_CASES = {'fixture-note-read', 'fixture-note-denied', 'fixture-note-cancelled',
          'fixture-note-heldout', 'fixture-note-restart', 'fixture-question-unavailable'}
_ACTOR = 'fixture-user'
_FILE = 'Receta.md'


def _only_case(case):
    if case.get('id') not in _CASES:
        raise AdapterUnavailable('synthetic_case_not_supported')
    return case['fixture']


def _v(actions=0, calls=0, input_tokens=0, output_tokens=0):
    return {'actions': actions, 'model_calls': calls,
            'input_tokens': input_tokens, 'output_tokens': output_tokens}


class SingleLoopFixtureAdapter:
    """Execute #214 with synthetic script, source, authorization and read tool."""
    offline = True
    revision = 'working-tree:single-loop-candidate:scripted-fixture-v1'

    def __call__(self, case, fixture_root, emit):
        from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
        from alphonse.agent_v2.core.intelligence.task_state import TaskState
        from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateHost, CandidateState, run_candidate
        from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import BudgetLimits, SQLiteTaskBudgetStore, TaskBudget
        from alphonse.agent_v2.core.messages import InMemoryMessageQueue
        from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition

        _only_case(case)
        fixture = case['fixture']
        source = {'goal': case['goal']}
        scripts = iter(case['provider_script']['candidate'])
        registry = InMemoryToolRegistry()
        descriptor = ToolDescriptor('native.fixture_read', 'fixture_read', ToolKind.NATIVE,
            argument_schema={'type': 'object', 'properties': {'path': {'type': 'string'}},
                             'required': ['path'], 'additionalProperties': False}, read_only=True)
        def read(args):
            emit({'kind': 'tool', 'tool': 'native.fixture_read', 'arguments': dict(args)})
            # No arbitrary filesystem paths and no synthetic external side effects.
            if args != {'path': _FILE}:
                raise PermissionError('fixture_read_denied')
            emit({'kind': 'source_read', 'source': _FILE})
            return {'candidate_complete': True, 'content': (fixture_root / _FILE).read_text()}
        registry.register(ToolDefinition(descriptor, read))
        task = TaskState(task_id='paired-fixture', user=_ACTOR)
        context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry,
                                  cancellation_checker=lambda: fixture.get('cancelled', False))
        checkpoints = []
        class FixtureInterrupted(BaseException):
            # Simulate abrupt process loss, which ordinary host exception
            # handling cannot convert into an evidence-recording failure.
            pass
        def choose(text, cap):
            decision = dict(next(scripts))
            # These are deterministic synthetic accounting units, never provider usage.
            output_units = len(json.dumps(decision, ensure_ascii=False).encode('utf-8'))
            if output_units > cap:
                raise ValueError('fixture_output_exceeds_cap')
            units = {'input_tokens': len(text.encode('utf-8')), 'output_tokens': output_units}
            emit({'kind': 'system_two', 'estimated_usage': units})
            if decision.get('kind') == 'question':
                # This is an attempted question, not proof of user delivery.
                emit({'kind': 'question'})
            decision['usage'] = units
            return decision
        def record(actor, result):
            if fixture.get('interrupt_before_evidence'):
                # The read has returned, but the observation has not yet
                # been accepted into the synthetic evidence store.
                raise FixtureInterrupted()
            ref = 'read:fixture'
            source[ref] = json.dumps(result['result'], ensure_ascii=False)
            return ref
        def verify(actor, text, refs):
            return (actor == _ACTOR and refs == ('read:fixture',)
                    and text == fixture.get('expected_response', 'La nota dice amoxicillin.')
                    and fixture.get('expected_content', 'Medicine: amoxicillin') in source[refs[0]])
        with TemporaryDirectory(prefix='alphonse-candidate-budget-') as directory:
            store = SQLiteTaskBudgetStore(str(Path(directory) / 'budget.db'))
            store.create(task.task_id, TaskBudget(BudgetLimits(_v(4, 5, 10000, 1200),
                         _v(0, 1, 300, 100), 100000, 1000), started_at_ms=0))
            checkpoint_path = Path(directory) / 'state.json'
            def checkpoint(snapshot):
                checkpoints.append(snapshot)
                checkpoint_path.write_text(json.dumps(snapshot), encoding='utf-8')
            host = CandidateHost(task, context, store, checkpoint, choose,
                lambda actor, ref: actor == _ACTOR and ref in source,
                lambda actor, ref: source[ref],
                lambda actor, tool, args: (fixture.get('permissions') == ['read'] and actor == _ACTOR
                                          and tool == 'native.fixture_read' and args == {'path': _FILE}),
                record, verify, lambda: 100, input_reservation=2048, output_reservation=256,
                projection_budget=2048)
            try:
                state = run_candidate(CandidateState(task.task_id, _ACTOR, 'goal'), host)
            except FixtureInterrupted:
                # Reconstruct both collaborators from persisted artifacts,
                # as a fresh worker would. This is a process-restart fixture,
                # not a power-failure or transactional durability test.
                restored = CandidateState.restore(json.loads(checkpoint_path.read_text(encoding='utf-8')))
                emit({'kind': 'restart', 'inflight': bool(restored.inflight)})
                reopened = SQLiteTaskBudgetStore(str(Path(directory) / 'budget.db'))
                state = run_candidate(restored, replace(host, budgets=reopened))
        return Run(status=state.status, response=state.response, reason=state.blocker or None,
                   measured_kinds=('system_two', 'jev', 'tool', 'question', 'source_read', 'restart'))
