"""Bounded independent read sequences still use the real tactical guards."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import PhaseExecutor, PhasePlan, PhaseSubgoal, SideEffectClass, TacticalState, new_tactical_state
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from alphonse.agent_v2.system_one import SystemOneTacticalReview

QUESTION = {'question_id': 'result', 'type': 'noul', 'instructions': 'Does the result meet the stage?',
            'criteria': {'true': 'The stage is met.', 'false': 'The stage remains incomplete.'}}


def action(key, tool='native.read'):
    return {'tool_id': tool, 'arguments': {'key': key}, 'acceptance_questions': [QUESTION]}


class Inference:
    def __init__(self, *values):
        self.values = iter(values)
        self.requests = []
    def generate_json(self, request):
        self.requests.append(request)
        return SimpleNamespace(json_value=next(self.values))


def setup(*, inference, probability=.99, callback=None, schema=None):
    calls = []
    registry = InMemoryToolRegistry()
    def read(arguments):
        calls.append(arguments['key'])
        if callback:
            callback(registry, calls)
        return {'key': arguments['key']}
    registry.register(ToolDefinition(ToolDescriptor('native.read', 'read', ToolKind.NATIVE, read_only=True, argument_schema=schema or {}), read))
    subgoal = PhaseSubgoal('stage', 'Read three independent records', 'records',
                           allowed_capabilities=('native.read',), allowed_side_effects=(SideEffectClass.READ_ONLY,))
    state = new_tactical_state(PhasePlan('phase', 'Read records', (subgoal,), authorized_capabilities=('native.read',)))
    state.revealed_tool_ids = ['native.read']
    def review(**_):
        done = len(calls) == 3
        return SystemOneTacticalReview(done, .99, True, answers={'runtime_action_added_progress': probability,
            'runtime_declared_subgoal_complete': .99 if done else .01})
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, inference=inference,
                              system_one=SimpleNamespace(evaluate_tactical_progress=review))
    return calls, registry, state, context


def test_independent_reads_save_two_composition_calls_with_the_same_observations():
    seq = Inference({'actions': [action('a'), action('b'), action('c')]})
    calls, _, state, context = setup(inference=seq)
    outcome = PhaseExecutor().run(TaskState(user='fixture'), state, context)
    assert outcome.status.value == 'phase_complete'
    assert calls == ['a', 'b', 'c']
    assert len(seq.requests) == 1
    assert len(state.actions) == 3


def test_uncertain_progress_discards_remaining_proposals_and_recomposes():
    seq = Inference({'actions': [action('a'), action('b'), action('c')]}, action('d'), action('e'))
    calls, _, state, context = setup(inference=seq, probability=.5)
    outcome = PhaseExecutor().run(TaskState(user='fixture'), state, context)
    assert outcome.status.value == 'phase_complete'
    assert calls == ['a', 'd', 'e']
    assert len(seq.requests) == 3


@pytest.mark.parametrize('invalid', [
    {'actions': [action('a'), action('a')]},
    {'actions': [action('a'), action('b'), action('c'), action('d')]},
    {'actions': [action('a'), action('x', 'native.write')]},
    {'actions': [action('a'), {'tool_id': 'native.read', 'arguments': {'key': 'secret'}}]},
])
def test_sequence_is_rejected_before_any_action_and_does_not_log_proposals(invalid):
    seq = Inference(invalid, None)
    calls, registry, state, context = setup(inference=seq)
    registry.register(ToolDefinition(ToolDescriptor('native.write', 'write', ToolKind.NATIVE, read_only=False), lambda _: {}))
    outcome = PhaseExecutor().run(TaskState(user='fixture'), state, context)
    assert outcome.status.value == 'blocked'
    assert calls == []
    assert state.evidence.entries[0]['rejected_action']['arguments'] == {'redacted': True}
    assert 'secret' not in str(state.evidence.entries)


def test_cancellation_stops_before_the_next_read():
    seq = Inference({'actions': [action('a'), action('b'), action('c')]})
    calls, _, state, context = setup(inference=seq)
    context.cancellation_checker = lambda: bool(calls)
    outcome = PhaseExecutor().run(TaskState(user='fixture'), state, context)
    assert outcome.status.value == 'cancelled'
    assert calls == ['a']


def test_revoked_tool_is_rechecked_before_the_next_read():
    def revoke(registry, calls):
        if len(calls) == 1:
            registry.register(replace(registry.get_definition('native.read'), enabled=False))
    seq = Inference({'actions': [action('a'), action('b'), action('c')]}, None)
    calls, _, state, context = setup(inference=seq, callback=revoke)
    outcome = PhaseExecutor().run(TaskState(user='fixture'), state, context)
    assert outcome.status.value == 'blocked'
    assert calls == ['a']


def test_restart_discards_unexecuted_sequence_instead_of_replaying_it():
    class Crash(BaseException):
        pass
    class Store:
        saved = None
        def save_task_checkpoint(self, task, **_):
            self.saved = task.to_dict()
            rows = task.hierarchical_state.get('evidence', {}).get('entries', [])
            if any(row.get('progress_review') for row in rows):
                raise Crash()
    seq = Inference({'actions': [action('a'), action('b'), action('c')]}, action('d'), action('e'))
    calls, _, state, context = setup(inference=seq)
    store = Store()
    context.question_store = store
    with pytest.raises(Crash):
        PhaseExecutor().run(TaskState(user='fixture'), state, context)
    assert calls == ['a']
    restored = TaskState.from_dict(store.saved)
    state = TacticalState.from_dict(restored.hierarchical_state)
    context.question_store = None
    outcome = PhaseExecutor().run(restored, state, context)
    assert outcome.status.value == 'phase_complete'
    assert calls == ['a', 'd', 'e']
    assert len(seq.requests) == 3


def test_ambiguous_acceptance_discards_remaining_reads_even_if_progress_is_positive():
    seq = Inference({'actions': [action('a'), action('b'), action('c')]}, action('d'), action('e'))
    calls, _, state, context = setup(inference=seq)
    def review(**_):
        done = len(calls) == 3
        return SystemOneTacticalReview(done, .6, done, answers={'runtime_action_added_progress': .95,
            'runtime_declared_subgoal_complete': .99 if done else .5})
    context.system_one.evaluate_tactical_progress = review
    outcome = PhaseExecutor().run(TaskState(user='fixture'), state, context)
    assert outcome.status.value == 'phase_complete'
    assert calls == ['a', 'd', 'e']
    assert len(seq.requests) == 3


def test_stage_completion_stops_the_remaining_sequence():
    seq = Inference({'actions': [action('a'), action('b'), action('c')]})
    calls, _, state, context = setup(inference=seq)
    context.system_one.evaluate_tactical_progress = lambda **_: SystemOneTacticalReview(
        True, .99, True, answers={'runtime_action_added_progress': .99, 'runtime_declared_subgoal_complete': .99})
    outcome = PhaseExecutor().run(TaskState(user='fixture'), state, context)
    assert outcome.status.value == 'phase_complete'
    assert calls == ['a']


def test_incomplete_evidence_discards_remaining_reads():
    seq = Inference({'actions': [action('a'), action('b'), action('c')]}, action('d'), action('e'))
    calls, _, state, context = setup(inference=seq)
    def review(**_):
        done = len(calls) == 3
        return SystemOneTacticalReview(done, .99, True, answers={'runtime_action_added_progress': .99,
            'runtime_declared_subgoal_complete': .99 if done else .01}, context_complete=len(calls) != 1)
    context.system_one.evaluate_tactical_progress = review
    outcome = PhaseExecutor().run(TaskState(user='fixture'), state, context)
    assert outcome.status.value == 'phase_complete'
    assert calls == ['a', 'd', 'e']
    assert len(seq.requests) == 3


def test_private_browser_sequence_is_rejected_without_logging_its_arguments():
    private = action('synthetic-secret', 'native.browser_workflow_observe')
    seq = Inference({'actions': [action('a'), private]}, None)
    calls, registry, state, context = setup(inference=seq)
    registry.register(ToolDefinition(ToolDescriptor('native.browser_workflow_observe', 'browser',
        ToolKind.NATIVE, read_only=True, metadata={'sensitive_content': True}), lambda _: {}))
    state.revealed_tool_ids.append('native.browser_workflow_observe')
    outcome = PhaseExecutor().run(TaskState(user='fixture'), state, context)
    assert outcome.status.value == 'blocked'
    assert calls == []
    assert 'synthetic-secret' not in str(state.evidence.entries)


def test_sequence_uses_distinct_runtime_action_and_evidence_ids():
    proposed = [dict(action(key), action_id='same-model-id') for key in ('a', 'b', 'c')]
    seq = Inference({'actions': proposed})
    calls, _, state, context = setup(inference=seq)
    outcome = PhaseExecutor().run(TaskState(user='fixture'), state, context)
    assert outcome.status.value == 'phase_complete'
    assert len({item.action_id for item in state.actions}) == 3
    refs = [entry['evidence_ref'] for entry in state.evidence.entries if entry.get('status') == 'success']
    assert len(refs) == len(set(refs)) == 3


def test_later_invalid_tool_arguments_reject_the_whole_sequence():
    malformed = action('b')
    malformed['arguments'] = {'wrong': 'synthetic-secret'}
    seq = Inference({'actions': [action('a'), malformed]}, None)
    schema = {'type': 'object', 'properties': {'key': {'type': 'string'}}, 'required': ['key']}
    calls, _, state, context = setup(inference=seq, schema=schema)
    outcome = PhaseExecutor().run(TaskState(user='fixture'), state, context)
    assert outcome.status.value == 'blocked'
    assert calls == []
    assert 'synthetic-secret' not in str(state.evidence.entries)


def test_later_reserved_jev_question_rejects_the_whole_sequence():
    malformed = action('b')
    malformed['acceptance_questions'] = [dict(QUESTION, question_id='runtime_action_added_progress')]
    seq = Inference({'actions': [action('a'), malformed]}, None)
    calls, _, state, context = setup(inference=seq)
    outcome = PhaseExecutor().run(TaskState(user='fixture'), state, context)
    assert outcome.status.value == 'blocked'
    assert calls == []


def test_nested_sequences_are_rejected_before_any_read():
    nested = dict(action('b'), actions=[action('c'), action('d')])
    seq = Inference({'actions': [action('a'), nested]}, None)
    calls, _, state, context = setup(inference=seq)
    outcome = PhaseExecutor().run(TaskState(user='fixture'), state, context)
    assert outcome.status.value == 'blocked'
    assert calls == []


def test_failed_first_read_stops_the_remaining_sequence():
    def fail(_registry, _calls):
        raise RuntimeError('synthetic source unavailable')
    seq = Inference({'actions': [action('a'), action('b'), action('c')]})
    calls, _, state, context = setup(inference=seq, callback=fail)
    outcome = PhaseExecutor().run(TaskState(user='fixture'), state, context)
    assert outcome.status.value == 'blocked'
    assert calls == ['a']
    assert state.evidence.entries[-1]['status'] == 'failed'
    assert not state.completed_subgoal_ids
