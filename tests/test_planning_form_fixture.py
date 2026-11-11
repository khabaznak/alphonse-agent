from copy import deepcopy

import pytest

from alphonse.agent_v2.evaluation.planning_form_fixture import FixtureDenied, FormFixture


def test_multifield_completion_requires_correct_values_and_readback():
    fixture = FormFixture()
    form = fixture.inspect('fixture-owner')
    fixture.fill('fixture-owner', form['version'], 'name', 'Synthetic Robin')
    with pytest.raises(FixtureDenied, match='incomplete_or_wrong_values'):
        fixture.submit('fixture-owner', form['version'])
    fixture.fill('fixture-owner', form['version'], 'code', 'fixture-17')
    fixture.submit('fixture-owner', form['version'])
    assert not fixture.oracle()['verified']
    assert fixture.verify('fixture-owner')['submitted'] == fixture.fields
    assert fixture.oracle() == {'verified': True, 'submission_count': 1, 'expected_values': True}
    with pytest.raises(FixtureDenied, match='effect_requires_reconciliation'):
        fixture.submit('fixture-owner', form['version'])


def test_changed_forms_denial_cancellation_and_restart_do_not_bypass_checks():
    fixture = FormFixture()
    version = fixture.inspect('fixture-owner')['version']
    fixture.fill('fixture-owner', version, 'name', 'Synthetic Robin')
    fixture.change_form({'name': 'Synthetic Robin', 'reference': 'fixture-17'})
    restored = FormFixture.restore(fixture.snapshot())
    with pytest.raises(FixtureDenied, match='stale_form'):
        restored.fill('fixture-owner', version, 'code', 'fixture-17')
    with pytest.raises(FixtureDenied, match='access_denied'):
        restored.inspect('other')
    restored.cancelled = True
    with pytest.raises(FixtureDenied, match='cancelled'):
        restored.fill('fixture-owner', restored.version, 'name', 'Synthetic Robin')
    assert restored.oracle()['submission_count'] == 0


def test_uncertain_external_effect_is_not_replayed_after_restart():
    fixture = FormFixture()
    version = fixture.inspect('fixture-owner')['version']
    for key, value in fixture.fields.items():
        fixture.fill('fixture-owner', version, key, value)
    assert fixture.submit('fixture-owner', version, ambiguous=True)['status'] == 'uncertain'
    restored = FormFixture.restore(deepcopy(fixture.snapshot()))
    with pytest.raises(FixtureDenied, match='effect_requires_reconciliation'):
        restored.submit('fixture-owner', version)
    assert restored.verify('fixture-owner')['submitted'] == restored.fields
    assert restored.oracle()['verified']
    assert restored.oracle()['submission_count'] == 1


def test_readback_before_submission_does_not_verify_a_later_effect():
    fixture = FormFixture()
    fixture.verify('fixture-owner')
    for key, value in fixture.fields.items():
        fixture.fill('fixture-owner', fixture.version, key, value)
    fixture.submit('fixture-owner', fixture.version)
    assert not fixture.oracle()['verified']


def test_real_tactical_executor_runs_multifield_fixture_with_verified_readback():
    from types import SimpleNamespace
    from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3 import PhaseExecutor, PhasePlan, PhaseSubgoal, SideEffectClass, new_tactical_state
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
    from alphonse.agent_v2.system_one import SystemOneTacticalReview

    fixture = FormFixture()
    registry = InMemoryToolRegistry()
    def run(args):
        operation = args['operation']
        if operation == 'inspect':
            return fixture.inspect('fixture-owner')
        if operation == 'fill':
            return fixture.fill('fixture-owner', args['version'], args['field'], args['value'])
        if operation == 'submit':
            return fixture.submit('fixture-owner', args['version'])
        if operation == 'verify':
            return fixture.verify('fixture-owner')
        raise FixtureDenied('invalid_operation')
    registry.register(ToolDefinition(ToolDescriptor('native.fixture_form', 'fixture_form', ToolKind.NATIVE,
        read_only=False, metadata={'side_effect_class': 'external_irreversible'}), run))
    subgoal = PhaseSubgoal('form', 'Complete and independently verify the fixture form', 'receipt',
        allowed_capabilities=('native.fixture_form',),
        allowed_side_effects=(SideEffectClass.READ_ONLY, SideEffectClass.EXTERNAL_IRREVERSIBLE))
    state = new_tactical_state(PhasePlan('fixture', 'Complete fixture form', (subgoal,),
                                        authorized_capabilities=('native.fixture_form',)))
    state.revealed_tool_ids = ['native.fixture_form']
    steps = iter([
        {'operation': 'inspect'},
        {'operation': 'fill', 'version': 1, 'field': 'name', 'value': 'Synthetic Robin'},
        {'operation': 'fill', 'version': 1, 'field': 'code', 'value': 'fixture-17'},
        {'operation': 'submit', 'version': 1},
        {'operation': 'verify'},
    ])
    def selector(*_):
        return {'tool_id': 'native.fixture_form', 'arguments': next(steps)}
    def review(**values):
        done = values['action']['arguments']['operation'] == 'verify' and fixture.oracle()['verified']
        return SystemOneTacticalReview(done, .99, True, answers={'runtime_declared_subgoal_complete': .99 if done else .01,
                                                                 'runtime_action_added_progress': .99})
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry,
                              system_one=SimpleNamespace(evaluate_tactical_progress=review))
    outcome = PhaseExecutor(action_selector=selector).run(TaskState(user='fixture-owner'), state, context)
    assert outcome.status.value == 'phase_complete'
    assert fixture.oracle() == {'verified': True, 'submission_count': 1, 'expected_values': True}
    assert [event['operation'] for event in fixture.events] == ['inspect', 'fill', 'fill', 'submit', 'verify']
