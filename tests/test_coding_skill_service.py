from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.coding_jobs import CodingJobStore
from alphonse.agent_v2.coding_skill_service import CodingSkillService
from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.invocation import ToolInvocationService
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.core.intelligence.v3 import PhaseExecutor, PhasePlan, PhaseStatus, PhaseSubgoal, SideEffectClass, new_tactical_state
from alphonse.agent_v2.system_one import SystemOneTacticalReview


class FixtureProvider:
    def __init__(self, *, invalid=False):
        self.calls = 0
        self.invalid = invalid

    def run(self, *, workspace, name, brief, cancelled):
        self.calls += 1
        package = workspace / name
        package.mkdir()
        (package / 'SKILL.md').write_text(f'---\nname: {name}\ndescription: Inspect fixture readings.\n---\nUse scripts/read.py to read the fixture.\n')
        (package / 'scripts').mkdir()
        (package / 'scripts/read.py').write_text("print('FIXTURE_OK')\n")
        if self.invalid:
            (package / 'data').mkdir()
        return 'completed'


class FixtureVerifier:
    def verify(self, *, package, checks, cancelled):
        result = subprocess.run([sys.executable, str(package / 'scripts/read.py')], capture_output=True, text=True, timeout=5)
        return result.returncode == 0 and result.stdout.strip() == 'FIXTURE_OK'


def service(tmp_path, provider=None, verifier=None, is_admin=lambda actor: actor == 'owner'):
    root = tmp_path / 'jobs'
    root.mkdir()
    return CodingSkillService(jobs=CodingJobStore(tmp_path / 'jobs.sqlite'),
        skills=SkillStore(tmp_path / 'skills'), workspace_root=root,
        provider=provider, verifier=verifier, is_admin=is_admin)


def test_fixture_provider_creates_verified_skill_that_can_be_used_and_replayed(tmp_path):
    provider = FixtureProvider()
    coordinator = service(tmp_path, provider, FixtureVerifier())
    request = dict(actor='owner', task_id='task', operation_id='one', name='fixture-reader', brief='Build the fixture reader', checks=['fixture-check'])
    result = coordinator.build(**request)
    assert result['skill_id'] == 'skill:fixture-reader'
    skill = coordinator.skills.get(result['skill_id'])
    assert 'scripts/read.py' in skill.instructions
    task = TaskState(user='owner')
    task.metadata['v3_skill_selection'] = {'selected_skill_ids': [skill.skill_id]}
    context = CoreLoopContext(messages=InMemoryMessageQueue(), skill_store=coordinator.skills, tools=build_native_tool_registry())
    outcome = ToolInvocationService(context=context, task=task).invoke('native.bash', {
        'skill_id': skill.skill_id, 'runtime': sys.executable,
        'command': '"$ALPHONSE_SKILL_RUNTIME" scripts/read.py',
    })
    assert outcome['status'] == 'success' and outcome['result']['stdout'].strip() == 'FIXTURE_OK'
    assert coordinator.build(**request)['replayed'] is True
    assert provider.calls == 1
    (Path(skill.directory) / 'scripts/read.py').write_text("print('changed')\n")
    with pytest.raises(RuntimeError, match='reconciliation_required'):
        coordinator.build(**request)


def test_native_tool_is_only_registered_for_explicitly_configured_service(tmp_path):
    coordinator = service(tmp_path, FixtureProvider(), FixtureVerifier())
    registry = build_native_tool_registry(skill_store=coordinator.skills, coding_skill_service=coordinator)
    assert any(tool.tool_id == 'native.coding_skill_build' for tool in registry.list())
    task = TaskState(user='owner', task_id='task')
    context = CoreLoopContext(messages=InMemoryMessageQueue(), skill_store=coordinator.skills, tools=registry)
    result = ToolInvocationService(context=context, task=task).invoke('native.coding_skill_build', {
        'name': 'fixture-reader', 'operation_id': 'one', 'brief': 'Build the fixture reader', 'checks': ['fixture-check'],
    })
    assert result['status'] == 'success' and result['result']['skill_id'] == 'skill:fixture-reader'
    assert result['result']['activated_for_task'] is True
    assert 'skill:fixture-reader' in task.metadata['v3_skill_selection']['selected_skill_ids']
    followup = ToolInvocationService(context=context, task=task).invoke('native.bash', {
        'skill_id': 'skill:fixture-reader', 'runtime': sys.executable,
        'command': '"$ALPHONSE_SKILL_RUNTIME" scripts/read.py',
    })
    assert followup['status'] == 'success' and followup['result']['stdout'].strip() == 'FIXTURE_OK'
    without_service = build_native_tool_registry(skill_store=coordinator.skills)
    assert not any(tool.tool_id == 'native.coding_skill_build' for tool in without_service.list())
    with pytest.raises(ValueError, match='coding_skill_store_mismatch'):
        build_native_tool_registry(skill_store=SkillStore(tmp_path / 'other-skills'), coding_skill_service=coordinator)


def test_phase_builds_activates_and_uses_skill_for_original_task(tmp_path):
    coordinator = service(tmp_path, FixtureProvider(), FixtureVerifier())
    registry = build_native_tool_registry(skill_store=coordinator.skills, coding_skill_service=coordinator)
    task = TaskState(user='owner', task_id='task', goal='Read the fixture')
    phase = PhasePlan('build-and-read', 'Build skill then read fixture', (
        PhaseSubgoal('build', 'Build fixture skill', 'skill', allowed_capabilities=('native.coding_skill_build',),
                     allowed_side_effects=(SideEffectClass.LOCAL_SKILL_INSTALLATION,)),
        PhaseSubgoal('read', 'Read fixture', 'reading', depends_on=('build',),
                     allowed_capabilities=('native.bash',), allowed_side_effects=(SideEffectClass.READ_ONLY,)),
    ), authorized_capabilities=('native.coding_skill_build', 'native.bash'))
    state = new_tactical_state(phase)
    actions = iter([
        {'tool_id': 'native.coding_skill_build', 'arguments': {'name': 'fixture-reader', 'operation_id': 'one',
            'brief': 'Build fixture reader', 'checks': ['fixture-check']}},
        {'tool_id': 'native.bash', 'arguments': {'skill_id': 'skill:fixture-reader', 'runtime': sys.executable,
            'command': '"$ALPHONSE_SKILL_RUNTIME" scripts/read.py'}},
    ])
    reviewer = SimpleNamespace(evaluate_tactical_progress=lambda **kwargs: SystemOneTacticalReview(True, .99, True))
    context = CoreLoopContext(messages=InMemoryMessageQueue(), skill_store=coordinator.skills, tools=registry, system_one=reviewer)
    outcome = PhaseExecutor(action_selector=lambda *_: next(actions)).run(task, state, context)
    assert outcome.status == PhaseStatus.PHASE_COMPLETE
    assert state.actions[-1].result['stdout'].strip() == 'FIXTURE_OK'
    assert task.metadata['v3_skill_selection']['selected_skill_ids'] == ['skill:fixture-reader']


def test_runtime_refresh_retains_explicitly_configured_coding_service(tmp_path):
    from alphonse.agent_v2.runtime import build_runtime_host, refresh_runtime_artifacts
    coordinator = service(tmp_path, FixtureProvider(), FixtureVerifier())
    runtime = build_runtime_host(skill_store=coordinator.skills, coding_skill_service=coordinator)
    assert runtime.core.tools.get('native.coding_skill_build') is not None
    refresh_runtime_artifacts(runtime)
    assert runtime.core.tools.get('native.coding_skill_build') is not None


def test_provider_and_verification_are_required_before_reserving_work(tmp_path):
    coordinator = service(tmp_path)
    with pytest.raises(RuntimeError, match='isolated_provider_unavailable'):
        coordinator.build(actor='owner', task_id='task', operation_id='one', name='fixture-reader', brief='build', checks=['fixture-check'])
    with pytest.raises(PermissionError, match='manager_required'):
        coordinator.build(actor='other', task_id='task', operation_id='one', name='fixture-reader', brief='build', checks=['fixture-check'])


@pytest.mark.parametrize('problem', ['private', 'verification', 'cancelled', 'access_revoked'])
def test_failed_or_revoked_work_never_publishes(tmp_path, problem):
    provider = FixtureProvider(invalid=problem == 'private')
    class Verification:
        def verify(self, *, package, checks, cancelled):
            return problem != 'verification'
    calls = 0
    def authorized(actor):
        nonlocal calls
        calls += 1
        return problem != 'access_revoked' or calls == 1
    def cancelled():
        return problem == 'cancelled' and provider.calls > 0
    coordinator = service(tmp_path, provider, Verification(), is_admin=authorized)
    with pytest.raises((ValueError, RuntimeError, PermissionError)):
        coordinator.build(actor='owner', task_id='task', operation_id='one', name='fixture-reader', brief='build', checks=['fixture-check'], cancelled=cancelled)
    assert coordinator.skills.get('fixture-reader') is None


def test_package_cannot_change_during_verification(tmp_path):
    class ChangesPackage:
        def verify(self, *, package, checks, cancelled):
            (package / 'scripts/read.py').write_text("print('changed')\n")
            return True
    coordinator = service(tmp_path, FixtureProvider(), ChangesPackage())
    with pytest.raises(ValueError, match='verification_changed_package'):
        coordinator.build(actor='owner', task_id='task', operation_id='one', name='fixture-reader', brief='build', checks=['fixture-check'])
    assert coordinator.skills.get('fixture-reader') is None


def test_package_symlink_is_rejected_before_verification(tmp_path):
    class SymlinkProvider(FixtureProvider):
        def run(self, *, workspace, name, brief, cancelled):
            result = super().run(workspace=workspace, name=name, brief=brief, cancelled=cancelled)
            (workspace / name / 'scripts/link').symlink_to(tmp_path / 'outside')
            return result
    coordinator = service(tmp_path, SymlinkProvider(), FixtureVerifier())
    with pytest.raises(ValueError, match='symlink'):
        coordinator.build(actor='owner', task_id='task', operation_id='one', name='fixture-reader', brief='build', checks=['fixture-check'])
    assert coordinator.skills.get('fixture-reader') is None


def test_lost_publication_receipt_requires_reconciliation_and_no_replay(tmp_path, monkeypatch):
    provider = FixtureProvider()
    coordinator = service(tmp_path, provider, FixtureVerifier())
    transition = coordinator.jobs.transition
    def fail_receipt(**kwargs):
        if kwargs['state'] == 'completed':
            raise OSError('receipt unavailable')
        return transition(**kwargs)
    monkeypatch.setattr(coordinator.jobs, 'transition', fail_receipt)
    request = dict(actor='owner', task_id='task', operation_id='one', name='fixture-reader', brief='build', checks=['fixture-check'])
    with pytest.raises(RuntimeError, match='publication_reconciliation_required'):
        coordinator.build(**request)
    assert coordinator.skills.get('fixture-reader') is not None
    with pytest.raises(RuntimeError, match='job_reconciliation_required'):
        coordinator.build(**request)
    assert provider.calls == 1


def test_reconcile_stops_both_containers_without_replaying_or_claiming_success(tmp_path):
    stopped = []
    class Stoppable:
        def stop(self, key):
            stopped.append(key)
            return True
    coordinator = service(tmp_path, Stoppable(), Stoppable())
    job = coordinator.jobs.reserve(actor='owner', task_id='task', operation_id='one', request_hash='a' * 64)
    coordinator.jobs.transition(job_id=job['job_id'], actor='owner', task_id='task', expected='queued', state='running')
    with pytest.raises(PermissionError, match='manager_required'):
        coordinator.reconcile(actor='other', task_id='task', job_id=job['job_id'])
    result = coordinator.reconcile(actor='owner', task_id='task', job_id=job['job_id'])
    assert result['state'] == 'interrupted' and len(stopped) == 2
    assert coordinator.skills.get('fixture-reader') is None


def test_reconcile_keeps_job_running_if_container_removal_unconfirmed(tmp_path):
    class Unavailable:
        def stop(self, key): return False
    coordinator = service(tmp_path, Unavailable(), Unavailable())
    job = coordinator.jobs.reserve(actor='owner', task_id='task', operation_id='one', request_hash='a' * 64)
    coordinator.jobs.transition(job_id=job['job_id'], actor='owner', task_id='task', expected='queued', state='running')
    with pytest.raises(RuntimeError, match='reconciliation_required'):
        coordinator.reconcile(actor='owner', task_id='task', job_id=job['job_id'])
    assert coordinator.jobs.get(job_id=job['job_id'], actor='owner', task_id='task')['state'] == 'running'


def test_native_reconciliation_requires_owner_and_original_task(tmp_path):
    class Stoppable:
        def stop(self, key): return True
    coordinator = service(tmp_path, Stoppable(), Stoppable())
    job = coordinator.jobs.reserve(actor='owner', task_id='original', operation_id='one', request_hash='a' * 64)
    coordinator.jobs.transition(job_id=job['job_id'], actor='owner', task_id='original', expected='queued', state='running')
    registry = build_native_tool_registry(skill_store=coordinator.skills, coding_skill_service=coordinator)
    task = TaskState(user='owner', task_id='recovery')
    context = CoreLoopContext(messages=InMemoryMessageQueue(), skill_store=coordinator.skills, tools=registry)
    result = ToolInvocationService(context=context, task=task).invoke('native.coding_skill_build', {
        'operation': 'reconcile', 'job_id': job['job_id'], 'target_task_id': 'original',
    })
    assert result['status'] == 'success' and result['result']['state'] == 'interrupted'


def test_unconfirmed_provider_cleanup_preserves_running_job_for_reconciliation(tmp_path):
    class UncertainProvider:
        def run(self, **kwargs): raise RuntimeError('coding_container_reconciliation_required')
    coordinator = service(tmp_path, UncertainProvider(), FixtureVerifier())
    request = dict(actor='owner', task_id='task', operation_id='one', name='fixture-reader', brief='build', checks=['check'])
    with pytest.raises(RuntimeError, match='container_reconciliation_required'):
        coordinator.build(**request)
    with pytest.raises(RuntimeError, match='job_reconciliation_required'):
        coordinator.build(**request)
