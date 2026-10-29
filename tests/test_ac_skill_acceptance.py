"""Issue #40: real harness + ordinary disposable CLI; deterministic model doubles.

No household scripts, accounts, configurations, or physical effects are used.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceResult
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import (
    HierarchicalCAPDProcessor, PhaseExecutor, PhasePlan, PhaseStatus,
    TacticalState, new_tactical_state,
)
from alphonse.agent_v2.core.intelligence.v3.planner import plan_phase
from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
from alphonse.agent_v2.skill_bindings import SkillBindings
from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.system_one import (
    SystemOneAdmissionDecision, SystemOneActRecommendation, SystemOneReviewResult,
    SystemOneTacticalReview, SystemOneToolRegistrySelection,
)
from alphonse.agent_v2.users import V2UserStore

SKILL = 'skill:ac-fixture'
QUESTIONS = [{'question_id': 'observed', 'type': 'noul',
              'instructions': 'Does the observed result establish this stage?',
              'criteria': {'true': 'Direct evidence matches.', 'false': 'Missing, failed or mismatched.'}}]


@pytest.fixture
def household(tmp_path, monkeypatch):
    skills = SkillStore(tmp_path / 'skills with spaces')
    monkeypatch.setattr(skills, '_bundled_skills_dir', lambda: tmp_path / 'no-bundled')
    skills.install_directory(Path(__file__).parent / 'fixtures/skills/ac-fixture')
    users = V2UserStore(tmp_path / 'users.sqlite')
    users.onboard(display_name='Owner', user_id='owner', users_root=tmp_path / 'profiles')
    users.create_user(display_name='Member', user_id='member')
    users.create_user(display_name='Other', user_id='other')
    device = tmp_path / 'private state.json'
    device.write_text(json.dumps({'devices': {'fake-study': {
        'available': True, 'temperature': 26.5, 'unit': 'C', 'target': 25, 'writes': 0}}}))
    bindings = SkillBindings(SQLiteIntegrationStore(tmp_path / 'integrations.sqlite'), users, skills)
    for identity, kind, extra in [
        ('settings', 'configuration', {'configuration': {'rooms': {
            'study': ['fake-study'], 'office': ['fake-study'], 'bedroom': ['fake-study', 'fake-other']}}}),
        ('device', 'data', {'target': str(device)}),
    ]:
        bindings.configure(actor_id='owner', owner_id='member', binding_id=identity,
                           skill_id=SKILL, kind=kind, **extra)
        bindings.select(actor_id='member', skill_id=SKILL, name=identity, binding_id=identity)
    return SimpleNamespace(skills=skills, bindings=bindings, device=device, tmp=tmp_path)


class Jev:
    """Bounded decisions only; optimistic reviews test deterministic failure guards."""
    def classify_task_admission(self, **values):
        assert [s['id'] for s in values['skill_candidates']] == [SKILL]
        return SystemOneAdmissionDecision(False, .02, True, selected_skill_ids=(SKILL,), primary_skill_ids=(SKILL,))

    def curate_request_tools(self, **values):
        return SystemOneToolRegistrySelection(selected_tool_ids=tuple(t.tool_id for t in values['tools']))
    select_plan_tools = curate_request_tools

    def evaluate_tactical_progress(self, **values):
        return SystemOneTacticalReview(True, .99, True, answers={'observed': .99}, retry_approved=True)

    def evaluate(self, **values):
        refs = [e['evidence_ref'] for e in values['evidence']['entries'] if e.get('status') == 'success']
        return SystemOneReviewResult(tuple({'criterion_id': c['id'], 'status': 'satisfied',
                                           'evidence_refs': refs} for c in values['contract']['criteria']), ())

    def recommend_act(self, **values):
        return SystemOneActRecommendation('complete', .99, True, 'Synthetic verified outcome', answers={
            'continuation_is_worthwhile': .1, 'user_input_can_unblock': .1,
            'closure_explanation_is_warranted': .99})


def plan(operations, *, respond=False):
    goals = []
    for operation in ['usage', 'devices'] + operations + (['respond'] if respond else []):
        cap = ('user_response' if operation == 'respond' else 'skill_resources'
               if operation in ('usage', 'devices') else 'local_shell')
        effect = 'user_response' if operation == 'respond' else 'external_reversible' if operation == 'set' else 'read_only'
        goals.append({'subgoal_id': operation, 'objective': operation, 'required_output_type': operation,
                      'depends_on': [goals[-1]['subgoal_id']] if goals else [],
                      'allowed_capabilities': [cap], 'allowed_side_effects': [effect],
                      'completion': {'kind': 'field_equals', 'field': 'exit_code', 'expected': 0}
                      if operation not in ('respond', 'usage', 'devices')
                      else {'kind': 'output_present', 'output_type': operation}})
    return {'phase_id': 'climate', 'objective': 'Read fresh temperature and verify requested AC change',
            'criterion_ids': ['ac-1'], 'acceptance_criteria': ['Requested climate result is observed and reported'],
            'authorized_capabilities': ['skill_resources', 'local_shell', 'user_response'],
            'mutation_scope': {'allow_external_effects': 'set' in operations or respond}, 'subgoals': goals}


def action(operation, *, room='study'):
    if operation in ('usage', 'devices'):
        return {'tool_id': 'native.skill_read', 'arguments': {
            'skill_id': SKILL, 'path': f'references/{operation}.md'}, 'acceptance_questions': QUESTIONS}
    return {'tool_id': 'native.bash', 'arguments': {
        'skill_id': SKILL, 'binding_names': ['settings', 'device'], 'runtime': sys.executable,
        'command': f'"$ALPHONSE_SKILL_RUNTIME" scripts/ac.py {operation} --room {room}'
                   + (' --target 23' if operation in ('set', 'verify') else ''),
    }, 'acceptance_questions': QUESTIONS}


class SystemTwo:
    """Scripted plan/commands/wording. Provider quality is an opt-in live check."""
    def __init__(self, task, operations, respond=False):
        self.task, self.operations, self.respond = task, operations, respond
        self.requests = []

    def generate_markdown(self, request):
        self.requests.append(request)
        return InferenceResult(content='1.- [ ] Requested climate result is observed and reported', model_profile=request.model_profile)

    def generate_json(self, request):
        self.requests.append(request)
        assert 'Synthetic room climate' in request.prompt
        if request.purpose == InferencePurpose.PHASE_PLANNING:
            value = plan(self.operations, respond=self.respond)
        else:
            operation = request.metadata['subgoal_id']
            if operation == 'respond':
                # Use actual checkpointed tool evidence, never the fixture's source file.
                observed = next(a for a in reversed(self.task.hierarchical_state['actions'])
                                if a['subgoal_id'] in ('verify', 'temperature'))
                reading = json.loads(observed['result']['stdout'])
                message = (f"Study: {reading['temperature']} °{reading['unit']} from {reading['source']} "
                           f"at {reading['observed_at']}.")
                if reading['verified']:
                    message += f" AC target verified at {reading['target']} °{reading['unit']}."
                value = {'tool_id': 'native.respond', 'arguments': {'message': message}, 'acceptance_questions': QUESTIONS}
            else:
                value = action(operation)
        return InferenceResult(json_value=value, model_profile=request.model_profile)


def setup(f, operations, *, respond=False, identity='fresh'):
    task = TaskState(goal='Read the study temperature' + (' and set its AC to 23 C' if 'set' in operations else ''),
                     user='member', task_id=identity, message_id=identity,
                     metadata={'routing_disposition': 'pdca_task'})
    provider = SystemTwo(task, operations, respond)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), skill_store=f.skills,
                              skill_bindings=f.bindings, tools=build_native_tool_registry(skill_store=f.skills),
                              inference=provider, system_one=Jev())
    return task, context, provider


def admitted_phase(f, operations):
    task, context, provider = setup(f, operations)
    assert HierarchicalCAPDProcessor._admit_initial_human_task(task, context) is False
    state = new_tactical_state(plan_phase(task, context))
    return task, context, state


def run(task, context, state, *, room='study'):
    return PhaseExecutor(reveal_policy=ToolRevealPolicy(),
                         action_selector=lambda state, goal, tools: action(goal.subgoal_id, room=room)).run(task, state, context)


def device(f, **updates):
    data = json.loads(f.device.read_text())
    data['devices']['fake-study'].update(updates)
    f.device.write_text(json.dumps(data))
    return data['devices']['fake-study']


def receipt(state):
    """Portable attempt/evidence receipt, including failures and explicit call bounds."""
    return {'status': state.status.value, 'attempts': len(state.actions),
            'failures': sum(a.status != 'success' for a in state.actions),
            'automatic_shell_retry_budget': 0,
            'scenario_call_bound': len(state.phase.subgoals),
            'completed': list(state.completed_subgoal_ids),
            'evidence': state.evidence.to_dict()}


@pytest.mark.parametrize('operations', [['temperature'], ['temperature', 'set', 'verify']])
def test_fresh_conversations_complete_through_real_processor(household, operations):
    f = household
    for repeat in range(2):
        task, context, provider = setup(f, operations, respond=True, identity=f'fresh-{repeat}')
        before = datetime.now(timezone.utc)
        result = HierarchicalCAPDProcessor().process(task, context)
        assert result.status.value == 'completed'
        assert not task.project_id
        assert task.metadata['v3_skill_selection']['loaded_skill_ids'] == [SKILL]
        state = TacticalState.from_dict(task.hierarchical_state)
        readings = [json.loads(a.result['stdout']) for a in state.actions if a.subgoal_id == 'temperature']
        assert len(readings) == 1
        assert before <= datetime.fromisoformat(readings[0]['observed_at']) <= datetime.now(timezone.utc)
        message = task.metadata['prepared_user_response']['message']
        assert '26.5 °C from fake-study at ' in message
        assert ('AC target verified at 23.0 °C' in message) == ('set' in operations)
        report = receipt(state)
        assert report['attempts'] == len(operations) + 3 and report['failures'] == 0
        assert report['attempts'] <= report['scenario_call_bound']
        assert not any(t.tool_id.startswith('artifact.') for t in context.tools.list())
        if 'set' in operations:
            assert device(f)['writes'] == repeat + 1
        (f.tmp / f'{len(operations)}-{repeat}-receipt.json').write_text(json.dumps(report, default=str))


@pytest.mark.parametrize('room,changes,error', [
    ('bedroom', {}, 'ambiguous_room'), ('unknown', {}, 'unknown_room'),
    ('study', {'available': False}, 'device_unavailable'),
])
def test_bad_room_or_outage_never_claims_success_or_retries(household, room, changes, error):
    device(household, **changes)
    task, context, state = admitted_phase(household, ['temperature', 'set', 'verify'])
    outcome = run(task, context, state, room=room)
    assert outcome.status == PhaseStatus.BLOCKED
    assert receipt(state)['attempts'] == 3 and receipt(state)['failures'] == 1
    assert error in state.actions[-1].result['stdout']
    assert state.completed_subgoal_ids == ['usage', 'devices'] and device(household)['writes'] == 0
    assert not task.metadata.get('prepared_user_response')


def test_acknowledgement_without_actual_change_fails_verification(household):
    device(household, ignore_write=True)
    task, context, state = admitted_phase(household, ['set', 'verify'])
    assert run(task, context, state).status == PhaseStatus.BLOCKED
    assert state.completed_subgoal_ids == ['usage', 'devices', 'set']
    assert receipt(state)['attempts'] == 4 and receipt(state)['failures'] == 1
    assert 'state_mismatch' in state.actions[-1].result['stdout']
    assert device(household)['target'] == 25 and device(household)['writes'] == 1


def test_resume_completed_write_runs_only_missing_readback(household):
    task, context, state = admitted_phase(household, ['set', 'verify'])
    def review(**values):
        if values['subgoal']['subgoal_id'] == 'set':
            context.cancellation_checker = lambda: True
        return SystemOneTacticalReview(True, .99, True)
    context.system_one.evaluate_tactical_progress = review
    run(task, context, state)
    assert state.completed_subgoal_ids == ['usage', 'devices', 'set'] and device(household)['writes'] == 1
    restored = TacticalState.from_dict(json.loads(json.dumps(state.to_dict())))
    restored.status, restored.cancellation_pending = PhaseStatus.RUNNING, False
    context.cancellation_checker = None
    context.system_one = Jev()
    assert run(task, context, restored).status == PhaseStatus.PHASE_COMPLETE
    assert [a.subgoal_id for a in restored.actions] == ['usage', 'devices', 'set', 'verify']
    assert device(household)['writes'] == 1
    assert json.loads(restored.actions[-1].result['stdout'])['verified']


def test_lost_acknowledgement_requires_readback_before_recovery(household):
    device(household, fail_after_write=True)
    task, context, state = admitted_phase(household, ['set', 'verify'])
    assert run(task, context, state).status == PhaseStatus.BLOCKED
    assert receipt(state)['failures'] == 1 and device(household)['writes'] == 1
    restored = TacticalState.from_dict(state.to_dict())
    restored.status = PhaseStatus.RUNNING
    outcome = run(task, context, restored)
    assert outcome.status == PhaseStatus.BLOCKED
    assert 'reconciliation' in outcome.blockers[0]
    assert device(household)['writes'] == 1
    # Explicit recovery is read-only. The failed write is never relabeled success.
    recovery = new_tactical_state(PhasePlan.from_dict(plan(['verify'])))
    assert run(task, context, recovery).status == PhaseStatus.PHASE_COMPLETE
    assert json.loads(recovery.actions[-1].result['stdout'])['target'] == 23
    assert state.actions[-1].status == 'failed' and device(household)['writes'] == 1


@pytest.mark.parametrize('denial', ['other', 'revoked', 'missing'])
def test_configuration_access_is_rechecked_before_dispatch(household, denial):
    task, context, state = admitted_phase(household, ['temperature'])
    if denial == 'other':
        task.user = 'other'
    elif denial == 'revoked':
        household.bindings.set_access('settings', actor_id='member', enabled=False)
    else:
        household.device.unlink()
    outcome = run(task, context, state)
    assert outcome.status == PhaseStatus.BLOCKED
    assert state.actions[-1].status == 'failed' and state.completed_subgoal_ids == ['usage', 'devices']
    assert not task.metadata.get('prepared_user_response')


def test_authorized_alias_resolves_device_and_daemon_cwd_is_irrelevant(household, monkeypatch):
    task, context, state = admitted_phase(household, ['temperature'])
    monkeypatch.chdir(household.tmp)
    assert run(task, context, state, room='office').status == PhaseStatus.PHASE_COMPLETE
    assert json.loads(state.actions[-1].result['stdout'])['source'] == 'fake-study'
    assert device(household)['writes'] == 0


def test_cancelled_change_never_dispatches(household):
    task, context, state = admitted_phase(household, ['set', 'verify'])
    context.cancellation_checker = lambda: True
    assert run(task, context, state).status == PhaseStatus.CANCELLED
    assert not state.actions and device(household)['writes'] == 0


def test_missing_shell_capability_blocks_cli_dispatch(household):
    task, context, _ = admitted_phase(household, ['set', 'verify'])
    value = plan(['set', 'verify'])
    value['authorized_capabilities'] = ['skill_resources']
    for subgoal in value['subgoals'][2:]:
        subgoal['allowed_capabilities'] = []
    state = new_tactical_state(PhasePlan.from_dict(value))
    assert run(task, context, state).status == PhaseStatus.BLOCKED
    assert len(state.actions) == 2 and device(household)['writes'] == 0
