"""Issue #29: disposable household programs; no adopted artifact implementation."""
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceResult
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor, TacticalState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.core.tools.registry.native.skill_maintain import execute_skill_maintain
from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.system_one import (
    SystemOneAdmissionDecision, SystemOneActRecommendation, SystemOneReviewResult,
    SystemOneTacticalReview, SystemOneToolRegistrySelection,
)
from alphonse.agent_v2.users import V2UserStore

SKILL = 'skill:fixture'
OLD = '---\nname: fixture\ndescription: Synthetic household report\n---\nRun scripts/report.py with Python; --self-test uses synthetic inputs.\n'
NEW = OLD + '\nThe report prints the sum of two synthetic counts (5). See references/repair.md. No live-data binding required.\n'
BROKEN = 'print(missing_total)\n'
FIXED = 'print(sum([2, 3]))\n'
QUESTION = [{'question_id': 'done', 'type': 'noul', 'instructions': 'Is this verified?',
             'criteria': {'true': 'Observed requested result', 'false': 'Failed or missing evidence'}}]


@pytest.fixture
def household(tmp_path, monkeypatch):
    skills = SkillStore(tmp_path / 'household skills with spaces')
    bundled = tmp_path / 'bundled' / 'fixture'
    bundled.mkdir(parents=True)
    (bundled / 'SKILL.md').write_text(OLD)
    (bundled / 'scripts').mkdir()
    (bundled / 'scripts/report.py').write_text(BROKEN)
    helper = tmp_path / 'bundled' / 'diagnosis'
    helper.mkdir()
    (helper / 'SKILL.md').write_text('---\nname: diagnosis\ndescription: Diagnose Python failures\n---\nInspect stderr and script source. Repair undefined variables and verify with synthetic data.\n')
    monkeypatch.setattr(skills, '_bundled_skills_dir', lambda: bundled.parent)
    users = V2UserStore(tmp_path / 'users.sqlite')
    users.onboard(display_name='Owner', user_id='owner', users_root=tmp_path / 'profiles')
    users.create_user(display_name='Member', user_id='member')
    task = TaskState(goal='Repair the existing report skill', user='owner', task_id='repair')
    task.metadata['v3_skill_selection'] = {'selected_skill_ids': [SKILL]}
    core = CoreLoopContext(messages=InMemoryMessageQueue(), skill_store=skills)
    return SimpleNamespace(skills=skills, users=users, task=task, core=core, tmp=tmp_path)


def arguments(f, **changes):
    return {'skill_id': SKILL, 'operation': 'apply', 'operation_id': 'fix-1',
            'expected_revision': f.skills.management_status(SKILL)['local_revision'],
            'files': {'SKILL.md': NEW, 'scripts/report.py': FIXED,
                      'references/repair.md': 'Run Python scripts/report.py --self-test. Expected stdout: 5.',
                      'assets/example.json': '{"counts": [2, 3]}'},
            'runtime': sys.executable,
            'checks': ['test "$("$ALPHONSE_SKILL_RUNTIME" scripts/report.py --self-test)" = 5'], **changes}


def apply(f, args=None):
    return execute_skill_maintain(args or arguments(f), context=f.core.tool_execution_context(f.task),
                                 store=f.skills, is_admin=f.users.is_admin)


def test_batch_override_replay_conflict_and_recoverable_versions(household):
    f = household
    args = arguments(f)
    result = apply(f, args)
    assert result['published'] and result['exit_code'] == 0
    assert result['checks'][0]['exit_code'] == 0
    assert f.skills.read_file(SKILL, 'scripts/report.py') == FIXED
    status = f.skills.management_status(SKILL)
    assert status['override'] and status['customized'] and status['owner'] == 'owner'
    assert Path(result['recovery_directory'], 'fixture/scripts/report.py').read_text() == BROKEN
    assert apply(f, args)['replayed']
    with pytest.raises(ValueError, match='operation_conflict'):
        apply(f, {**args, 'checks': ['exit 1']})
    # Bundled upgrades stay separate and require explicit conflict resolution.
    (f.skills._bundled_skills_dir() / 'fixture/SKILL.md').write_text(OLD + '\nBundled upgrade')
    assert f.skills.management_status(SKILL)['bundled_update_available']
    with pytest.raises(ValueError, match='skill_update_conflict'):
        f.skills.replace_directory(SKILL, f.skills._bundled_skills_dir() / 'fixture')
    # Recovery is another verified resource edit, not a rollback of private data.
    recover = arguments(f, operation_id='restore', files={'SKILL.md': NEW, 'scripts/report.py': 'print(6)\n'},
                        checks=['test "$("$ALPHONSE_SKILL_RUNTIME" scripts/report.py)" = 6'])
    second = apply(f, recover)
    assert second['published']
    previous = Path(second['recovery_directory'], 'fixture/scripts/report.py').read_text()
    assert previous == FIXED
    restored = arguments(f, operation_id='restore-previous', files={'SKILL.md': NEW, 'scripts/report.py': previous})
    assert apply(f, restored)['published']
    assert f.skills.read_file(SKILL, 'scripts/report.py') == FIXED
    assert Path(result['recovery_directory'], 'fixture/scripts/report.py').read_text() == BROKEN


@pytest.mark.parametrize('changes,error', [
    ({'checks': ['echo synthetic failure >&2; exit 2']}, 'validation_failed'),
    ({'files': {'SKILL.md': 'broken'}}, 'frontmatter_required'),
    ({'files': {'SKILL.md': NEW, '../escape': 'x'}}, 'resource_path_required'),
    ({'files': {'SKILL.md': NEW, 'assets/secrets.json': 'x'}}, 'resource_path_required'),
    ({'checks': ['echo changed >> scripts/report.py']}, 'validation_changed_package'),
    ({'checks': ['sleep 1'], 'timeout_seconds': .02}, 'validation_failed'),
    ({'runtime': 'nonexistent-fixture-runtime'}, 'runtime_unavailable'),
])
def test_invalid_or_unverified_edits_never_activate_and_keep_drafts(household, changes, error):
    f = household
    before = f.skills.management_status(SKILL)['local_revision']
    args = arguments(f, **changes)
    result = apply(f, args)
    assert not result['published'] and result['exit_code'] == 1 and error in result['error']
    assert f.skills.management_status(SKILL)['local_revision'] == before
    assert list((f.skills.skills_dir / '.skill-state/fixture').glob('draft-*'))
    with pytest.raises(ValueError, match='reconciliation_required'):
        apply(f, args)


def test_validation_effects_not_repeated_after_failure_and_incomplete_journal(household):
    f = household
    counter = f.tmp / 'side effect count'
    args = arguments(f, checks=[f'echo once >> {shlex.quote(str(counter))}; exit 3'])
    assert not apply(f, args)['published']
    with pytest.raises(ValueError, match='reconciliation_required'):
        apply(f, args)
    assert counter.read_text() == 'once\n'
    journal = next((f.skills.skills_dir / '.skill-state/fixture').glob('operation-*'))
    state = json.loads(journal.read_text())
    state['status'] = 'started'  # A process died after dispatch and before its receipt.
    journal.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='reconciliation_required'):
        apply(f, args)
    assert counter.read_text() == 'once\n'


def test_private_data_bindings_and_credentials_survive_and_are_not_edit_targets(household):
    f = household
    f.skills.customize(SKILL, owner='owner')
    root = Path(f.skills.get(SKILL).directory)
    (root / 'data').mkdir()
    (root / 'data/household.sqlite').write_bytes(b'private household record')
    (root / '.env').write_text('TOKEN=synthetic-secret')
    args = arguments(f)
    assert apply(f, args)['published']
    assert (root / 'data/household.sqlite').read_bytes() == b'private household record'
    assert (root / '.env').read_text() == 'TOKEN=synthetic-secret'
    assert 'synthetic-secret' not in json.dumps(apply(f, args))
    for relative in ('data/household.sqlite', '.env', 'assets/private.key'):
        result = apply(f, arguments(f, operation_id=relative, files={'SKILL.md': NEW, relative: 'overwrite'}))
        assert not result['published']


def test_authority_selection_revocation_and_concurrent_change(household, monkeypatch):
    f = household
    f.task.user = 'member'
    with pytest.raises(PermissionError, match='manager_required'):
        apply(f)
    f.task.user = 'owner'
    f.task.metadata['v3_skill_selection']['selected_skill_ids'] = []
    with pytest.raises(PermissionError, match='not_selected'):
        apply(f)
    f.task.metadata['v3_skill_selection']['selected_skill_ids'] = [SKILL]
    import alphonse.agent_v2.core.tools.registry.native.skill_maintain as native
    original = native.execute_bash
    def concurrent(args, **kwargs):
        result = original(args, **kwargs)
        source = Path(f.skills.get(SKILL).directory) / 'SKILL.md'
        source.write_text(OLD + '\nConcurrent edit')
        return result
    monkeypatch.setattr(native, 'execute_bash', concurrent)
    result = apply(f)
    assert not result['published'] and 'conflict' in result['error']
    assert f.skills.read_file(SKILL, 'scripts/report.py') == BROKEN


def test_publication_exception_restores_active_package(household, monkeypatch):
    f = household
    f.skills.customize(SKILL, owner='owner')
    original = Path.rename
    def fail_publish(path, target):
        if (path.name == 'fixture' and path.parent.name.startswith('.skill-maintain-')
                and Path(target) == f.skills.skills_dir / 'fixture'):
            raise OSError('synthetic publication failure')
        return original(path, target)
    monkeypatch.setattr(Path, 'rename', fail_publish)
    result = apply(f)
    assert not result['published']
    assert 'synthetic publication failure' in result['error']
    assert f.skills.read_file(SKILL, 'scripts/report.py') == BROKEN
    assert list((f.skills.skills_dir / '.skill-state/fixture').glob('draft-*'))


def test_lost_publication_receipt_preserves_truth_and_never_repeats_checks(household, monkeypatch):
    f = household
    import alphonse.agent_v2.core.tools.registry.native.skill_maintain as native
    original = native.save_json
    count = f.tmp / 'validation-counter'
    args = arguments(f, checks=[f'echo once >> {shlex.quote(str(count))}'])
    def fail_receipt(path, value):
        if value.get('status') == 'published':
            raise OSError('synthetic receipt failure')
        return original(path, value)
    monkeypatch.setattr(native, 'save_json', fail_receipt)
    result = apply(f, args)
    assert result['published'] and result['exit_code'] == 1
    assert 'reconciliation_required' in result['exception']['code']
    assert f.skills.read_file(SKILL, 'scripts/report.py') == FIXED
    with pytest.raises(ValueError, match='reconciliation_required'):
        apply(f, args)
    assert count.read_text() == 'once\n'


def test_other_tasks_keep_their_pins_and_external_edits_fail_closed(household):
    from alphonse.agent_v2.core.intelligence.v3.context import selected_skill_guidance
    from alphonse.agent_v2.skill_resources import SkillActivationError
    f = household
    old_task = TaskState(goal='Use report', user='owner', task_id='other-task')
    old_task.metadata['v3_skill_selection'] = {'selected_skill_ids': [SKILL]}
    selected_skill_guidance(old_task, f.core)
    assert apply(f)['published']
    with pytest.raises(SkillActivationError, match='package_changed'):
        selected_skill_guidance(old_task, f.core)
    assert 'The report prints' in selected_skill_guidance(f.task, f.core)[0]


def test_concurrent_target_symlink_never_changes_external_storage(household, monkeypatch):
    f = household
    import alphonse.agent_v2.core.tools.registry.native.skill_maintain as native
    outside = f.tmp / 'external'
    outside.mkdir()
    (outside / 'sentinel').write_text('keep')
    original = native.execute_bash
    def introduce_link(args, **kwargs):
        result = original(args, **kwargs)
        (f.skills.skills_dir / 'fixture').symlink_to(outside, target_is_directory=True)
        return result
    monkeypatch.setattr(native, 'execute_bash', introduce_link)
    result = apply(f)
    assert not result['published'] and 'symlink' in result['error']
    assert (outside / 'sentinel').read_text() == 'keep'
    assert list(outside.iterdir()) == [outside / 'sentinel']
    assert (f.skills.skills_dir / 'fixture').is_symlink()


@pytest.mark.parametrize('caps,effects,allowed', [
    (['skill_management'], ['local_skill_installation'], False),
    (['local_shell'], ['local_skill_installation'], False),
    (['skill_management', 'local_shell'], ['read_only'], False),
    (['skill_management', 'local_shell'], ['local_skill_installation'], True),
])
def test_maintenance_reveal_requires_both_capabilities_and_install_effect(household, caps, effects, allowed):
    from alphonse.agent_v2.core.intelligence.v3 import PhasePlan, new_tactical_state
    from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy
    from alphonse.agent_v2.core.tools.registry.native.skill_maintain import build_skill_maintain_tool_definition
    f = household
    phase = PhasePlan.from_dict({'phase_id': 'maintenance', 'objective': 'Repair', 'criterion_ids': ['repair'],
                                'authorized_capabilities': caps, 'subgoals': [{
                                    'subgoal_id': 'edit', 'objective': 'Repair resources', 'required_output_type': 'edit',
                                    'allowed_capabilities': caps, 'allowed_side_effects': effects,
                                    'completion': {'kind': 'output_present', 'output_type': 'edit'}}]})
    descriptor = build_skill_maintain_tool_definition(f.skills, is_admin=f.users.is_admin).descriptor
    revealed = ToolRevealPolicy().reveal(f.task, new_tactical_state(phase), phase.subgoals[0], (descriptor,))
    assert bool(revealed.tools) is allowed


@pytest.mark.parametrize('interruption', ['cancelled', 'revoked'])
def test_interruption_after_validation_prevents_publication(household, monkeypatch, interruption):
    f = household
    import alphonse.agent_v2.core.tools.registry.native.skill_maintain as native
    original = native.execute_bash
    def interrupt(args, **kwargs):
        result = original(args, **kwargs)
        if interruption == 'cancelled':
            f.core.cancellation_checker = lambda: True
            # Execution context is immutable; cancellation reads this shared flag.
            cancelled[0] = True
        else:
            monkeypatch.setattr(f.skills, 'can_use', lambda *args: False)
        return result
    cancelled = [False]
    f.core.cancellation_checker = lambda: cancelled[0]
    monkeypatch.setattr(native, 'execute_bash', interrupt)
    result = apply(f)
    assert not result['published']
    assert ('cancelled' if interruption == 'cancelled' else 'access_revoked') in result['error']
    assert f.skills.read_file(SKILL, 'scripts/report.py') == BROKEN


def test_bindings_remain_private_and_revoked_after_maintenance(household):
    from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
    from alphonse.agent_v2.skill_bindings import SkillBindings
    f = household
    bindings = SkillBindings(SQLiteIntegrationStore(f.tmp / 'integrations.sqlite'), f.users, f.skills)
    data = f.tmp / 'private.sqlite'
    data.write_bytes(b'private household data')
    bindings.configure(actor_id='owner', owner_id='owner', binding_id='private', skill_id=SKILL,
                       kind='data', target=str(data))
    bindings.select(actor_id='owner', skill_id=SKILL, name='records', binding_id='private')
    assert apply(f)['published']
    assert bindings.prepare(actor_id='owner', skill_id=SKILL, names=['records']).values['records']['path'] == str(data)
    with pytest.raises(PermissionError):
        bindings.select(actor_id='member', skill_id=SKILL, name='records', binding_id='private')
    bindings.set_access('private', actor_id='owner', enabled=False)
    assert apply(f, arguments(f, operation_id='second'))['published']
    with pytest.raises((ValueError, PermissionError)):
        bindings.prepare(actor_id='owner', skill_id=SKILL, names=['records'])
    assert data.read_bytes() == b'private household data'


class Jev:
    def classify_task_admission(self, **values):
        ids = (SKILL, 'skill:diagnosis')
        return SystemOneAdmissionDecision(False, .01, True, selected_skill_ids=ids, primary_skill_ids=(SKILL,))

    def curate_request_tools(self, **values):
        return SystemOneToolRegistrySelection(selected_tool_ids=tuple(t.tool_id for t in values['tools']))
    select_plan_tools = curate_request_tools

    def evaluate_tactical_progress(self, **values):
        return SystemOneTacticalReview(True, .99, True, retry_approved=True)

    def evaluate(self, **values):
        refs = [e['evidence_ref'] for e in values['evidence']['entries'] if e.get('status') == 'success']
        return SystemOneReviewResult(tuple({'criterion_id': c['id'], 'status': 'satisfied', 'evidence_refs': refs}
                                          for c in values['contract']['criteria']), ())

    def recommend_act(self, **values):
        return SystemOneActRecommendation('complete', .99, True, 'Verified', answers={
            'continuation_is_worthwhile': .1, 'user_input_can_unblock': .1, 'closure_explanation_is_warranted': .99})


class SystemTwo:
    """Deterministic language/model double; processor, tools and scripts are real."""
    def __init__(self, f, repair):
        self.f, self.repair, self.requests = f, repair, []
        self.plans = 0

    def generate_markdown(self, request):
        if request.purpose == InferencePurpose.FINAL_RESPONSE:
            assert 'task that has already failed' in request.prompt
            if getattr(self.f, 'invalid_check', False):
                assert 'validation_failed' in request.prompt or 'reconciliation_required' in request.prompt
            return InferenceResult(content='The report repair could not be verified. The existing skill is preserved. Inspect the retained draft and check evidence before retrying.',
                                   model_profile=request.model_profile)
        return InferenceResult(content='1.- [ ] Requested report works and is verified', model_profile=request.model_profile)

    def generate_json(self, request):
        self.requests.append(request)
        if request.purpose == InferencePurpose.PHASE_PLANNING:
            self.plans += 1
            ops = ['diagnose', 'maintain', 'respond'] if self.repair else ['run', 'respond']
            if self.repair and self.plans > 1:
                ops = ['maintain', 'respond']  # Preserve completed diagnostic work.
            goals = []
            for op in ops:
                caps = ['skill_resources'] if op == 'diagnose' else ['skill_management', 'local_shell'] if op == 'maintain' else ['user_response'] if op == 'respond' else ['local_shell']
                goals.append({'subgoal_id': op, 'objective': op, 'required_output_type': op,
                              'depends_on': [goals[-1]['subgoal_id']] if goals else [], 'allowed_capabilities': caps,
                              'allowed_side_effects': ['local_skill_installation'] if op == 'maintain' else ['user_response'] if op == 'respond' else ['read_only'],
                              'completion': {'kind': 'field_equals', 'field': 'exit_code', 'expected': 0} if op in ('run', 'maintain') else {'kind': 'output_present', 'output_type': op}})
            value = {'phase_id': 'repair' if self.repair else 'use', 'objective': 'Requested report works and is verified',
                     'criterion_ids': ['report'], 'acceptance_criteria': ['Requested report works and is verified'],
                     'authorized_capabilities': ['skill_resources', 'skill_management', 'local_shell', 'user_response'],
                     'mutation_scope': {'allow_external_effects': True}, 'subgoals': goals}
        else:
            op = request.metadata['subgoal_id']
            if op == 'diagnose':
                assert 'Diagnose Python failures' in request.prompt
                value = {'tool_id': 'native.skill_read', 'arguments': {'skill_id': SKILL, 'path': 'scripts/report.py'}}
            elif op == 'maintain':
                if self.plans == 1:
                    assert 'missing_total' in request.prompt
                args = arguments(self.f)
                if getattr(self.f, 'invalid_check', False):
                    args['checks'] = ['echo unrecoverable-fixture-dependency >&2; exit 7']
                value = {'tool_id': 'native.skill_maintain', 'arguments': args}
            elif op == 'run':
                assert 'Synthetic household report' in request.prompt
                if self.f.skills.read_file(SKILL, 'scripts/report.py') == FIXED:
                    assert 'The report prints the sum of two synthetic counts (5)' in request.prompt
                value = {'tool_id': 'native.bash', 'arguments': {'skill_id': SKILL, 'runtime': sys.executable,
                         'command': '"$ALPHONSE_SKILL_RUNTIME" scripts/report.py --self-test'}}
            else:
                value = {'tool_id': 'native.respond', 'arguments': {'message': 'Report verified: 5.'}}
            value['acceptance_questions'] = QUESTION
        return InferenceResult(json_value=value, model_profile=request.model_profile)


def processor(f, identity, repair=False):
    task = TaskState(goal='Repair the report skill' if repair else 'Run the report skill', user='owner',
                     task_id=identity, message_id=identity, metadata={'routing_disposition': 'pdca_task'})
    provider = SystemTwo(f, repair)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), skill_store=f.skills, inference=provider,
                              system_one=Jev(), tools=build_native_tool_registry(skill_store=f.skills, user_store=f.users))
    result = HierarchicalCAPDProcessor().process(task, context)
    return result, task, context


def test_failure_diagnosis_repair_and_fresh_use_through_real_processor(household):
    f = household
    result, broken, _ = processor(f, 'broken')
    assert result.status.value != 'completed'
    state = TacticalState.from_dict(broken.hierarchical_state)
    assert state.actions[0].status == 'failed'
    assert 'NameError' in state.actions[0].result['stderr']
    result, repaired, context = processor(f, 'repair', repair=True)
    assert result.status.value == 'completed', json.dumps(repaired.metadata, default=str)
    state = TacticalState.from_dict(repaired.hierarchical_state)
    assert [a.tool_id for a in state.actions] == ['native.skill_read', 'native.skill_maintain', 'native.respond']
    assert state.actions[1].result['checks'][0]['exit_code'] == 0
    assert repaired.metadata['v3_skill_publications'][0]['from'] != repaired.metadata['v3_skill_publications'][0]['to']
    assert not any(t.tool_id.startswith('artifact.') for t in context.tools.list())
    result, fresh, _ = processor(f, 'fresh')
    assert result.status.value == 'completed' and not fresh.project_id
    assert fresh.metadata['v3_skill_selection']['loaded_skill_ids'] == [SKILL, 'skill:diagnosis']
    state = TacticalState.from_dict(fresh.hierarchical_state)
    assert state.actions[0].result['stdout'].strip() == '5'
    assert 'Report verified: 5' in fresh.metadata['prepared_user_response']['message']


def test_failed_validation_cannot_become_processor_success(household):
    f = household
    f.invalid_check = True
    result, task, _ = processor(f, 'failed-repair', repair=True)
    assert result.status.value != 'completed'
    state = TacticalState.from_dict(task.hierarchical_state)
    assert state.actions[-1].status == 'failed'
    evidence = [e for phase in task.metadata['v3_phase_history'] for e in phase['evidence']['entries']
                if isinstance(e.get('result'), dict) and e['result'].get('published') is False]
    assert len({e['evidence_ref'] for e in evidence}) == 1
    assert 'validation_failed' in evidence[0]['result']['error']
    assert not task.metadata.get('prepared_user_response')
    assert 'could not be verified' in task.metadata['v3_failure_response']['message']
    assert 'retained draft' in task.metadata['v3_failure_response']['message']
    assert f.skills.read_file(SKILL, 'scripts/report.py') == BROKEN
