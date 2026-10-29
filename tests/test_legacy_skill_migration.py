"""Disposable legacy programs only; never inspect or repair household resources."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import shlex
import sys

import pytest

from alphonse.agent_v2.artifacts import SQLiteArtifactStore
from alphonse.agent_v2.artifact_resources import ArtifactResources
from alphonse.agent_v2.core.core import CoreLoopContext, ToolExecutionContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.context import selected_skill_guidance
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.core.tools.registry.native.bash import execute_bash, BASH_ARGUMENT_SCHEMA
from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
from alphonse.agent_v2.legacy_migration import LegacyMigration
from alphonse.agent_v2.skill_bindings import SkillBindings
from alphonse.agent_v2.skill_resources import read_resource
from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture(params=['project', 'shared', 'skill'])
def legacy(tmp_path, request):
    users = V2UserStore(tmp_path / 'users.sqlite')
    users.onboard(display_name='Owner', user_id='owner', users_root=tmp_path / 'profiles')
    users.create_user(display_name='Member', user_id='member')
    users.create_user(display_name='Stranger', user_id='stranger')
    skills = SkillStore(tmp_path / 'skills')
    skill = skills.create_skill('fixture', 'Disposable migration fixture', 'Use documented commands.\n\n## Other workflow\nKEEP_OTHER_WORKFLOW\n')
    store = SQLiteArtifactStore(tmp_path / 'catalog.sqlite')
    projects = ProjectStore(tmp_path / 'projects.sqlite')
    project = projects.create_project(name='Legacy', owner_user_id='owner', root_path=str(tmp_path / 'legacy root with spaces'))
    resources = ArtifactResources(store, skills, projects, is_admin=users.is_admin)
    kind = request.param
    directory = resources.shared_directory(skill.skill_id, 'artifact.fixture') if kind == 'shared' else Path(skill.directory) if kind == 'skill' else Path(project.root_path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'reader.py').write_text("from pathlib import Path\nprint(Path('data.txt').read_text())\n")
    (directory / 'data.txt').write_text('original household data')
    (directory / 'config.env').write_text('SYNTHETIC_SECRET_NOT_FOR_PROMPTS')
    record = store.register(artifact_id='artifact.fixture', name='Fixture', description='Disposable reader',
        owner_user_id='owner', shared_with_user_ids=('member',), project_id=project.project_id,
        skill_id=skill.skill_id, storage_kind=kind, legacy_root_path=str(Path(project.root_path)),
        entrypoint_path='reader.py', argument_schema={'type': 'object'},
        execution={'runtime_path': sys.executable, 'required_files': ['data.txt'], 'configuration_paths': ['config.env']})
    skills.add_artifact_instructions(skill.skill_id, record.artifact_id, record.name, 'OLD_PRIVATE_INSTRUCTIONS\n\n## Private detail\nPRIVATE_LEGACY_SUBHEADING', record.skill_definition())
    # Nonlegacy sections survive; all headings inside legacy instructions are private.
    path = Path(skill.directory) / 'SKILL.md'
    before = path.read_bytes()
    integrations = SQLiteIntegrationStore(tmp_path / 'integrations.sqlite')
    bindings = SkillBindings(integrations, users, skills)
    migration = LegacyMigration(resources, bindings)
    return SimpleNamespace(**locals())


def preview(f):
    return f.migration.preview('owner')[0]


def apply(f, **kwargs):
    return f.migration.apply('owner', f.record.artifact_id, review_token=preview(f)['review_token'],
                             instructions='REVIEWED_PRIVATE_INSTRUCTIONS. ' + preview(f)['suggested_command'], **kwargs)


def context(f, actor='owner'):
    task = TaskState(user=actor, metadata={'v3_skill_selection': {'selected_skill_ids': [f.skill.skill_id]}})
    return ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), skill_store=f.skills,
                                artifact_resources=f.resources, skill_bindings=f.bindings)


def test_inventory_apply_no_project_execution_and_rollback(legacy, monkeypatch, tmp_path):
    f = legacy
    report = preview(f)
    assert report['source']['owner_user_id'] == 'owner'
    assert report['skill_id'] == f.skill.skill_id
    assert report['runtime_path'] == str(Path(sys.executable).absolute())
    assert report['configuration_paths'][0]['path'] == str(f.directory / 'config.env')
    assert 'SYNTHETIC_SECRET' not in json.dumps(report)
    assert report['file_moves'] == []
    result = apply(f)
    after = (Path(f.skill.directory) / 'SKILL.md').read_bytes()
    assert b'Artifact ID:' not in after and b'KEEP_OTHER_WORKFLOW' in after
    assert b'REVIEWED_PRIVATE' not in after  # Paths/instructions stay actor-scoped.
    assert f.store.get(f.record.artifact_id) == f.record  # Provenance unchanged.
    assert apply(f) == result
    assert (Path(f.skill.directory) / 'SKILL.md').read_bytes() == after
    monkeypatch.chdir(tmp_path)
    ctx = context(f)
    text, loaded = selected_skill_guidance(ctx.task, CoreLoopContext(messages=ctx.messages, skill_store=f.skills,
        artifact_resources=f.resources, skill_bindings=f.bindings))
    assert loaded == [f.skill.skill_id]
    assert 'REVIEWED_PRIVATE' in text and 'KEEP_OTHER_WORKFLOW' in text
    assert f.record.artifact_id not in text and 'artifact_id' not in text
    assert 'SYNTHETIC_SECRET' not in text and 'OLD_PRIVATE' not in text
    assert 'PRIVATE_LEGACY_SUBHEADING' not in text
    response = execute_bash({'skill_id': f.skill.skill_id, 'binding_names': [result['binding_name']],
        'command': report['suggested_command']}, context=ctx)
    assert response['exit_code'] == 0 and response['stdout'].strip() == 'original household data'
    assert f.migration.rollback('owner', f.record.artifact_id)['state'] == 'rolled_back'
    assert f.migration.rollback('owner', f.record.artifact_id)['state'] == 'rolled_back'
    assert (Path(f.skill.directory) / 'SKILL.md').read_bytes() == f.before
    assert (f.directory / 'data.txt').read_text() == 'original household data'
    assert (f.directory / 'reader.py').read_text().startswith('from pathlib')
    assert (f.directory / 'config.env').read_text() == 'SYNTHETIC_SECRET_NOT_FOR_PROMPTS'
    assert apply(f)['state'] == 'complete'


def test_access_and_binding_revocation(legacy):
    f = legacy
    assert f.migration.preview('stranger') == []
    with pytest.raises(PermissionError):
        f.migration.apply('stranger', f.record.artifact_id, review_token=preview(f)['review_token'], instructions='anything')
    result = apply(f)
    name = result['binding_name']
    assert f.bindings.prepare(actor_id='member', skill_id=f.skill.skill_id, names=[name]).values
    with pytest.raises(ValueError, match='choice_missing'):
        f.bindings.prepare(actor_id='stranger', skill_id=f.skill.skill_id, names=[name])
    assert 'REVIEWED_PRIVATE' not in f.migration.guidance(f.skill.skill_id, 'stranger')
    f.bindings.set_access(name, actor_id='owner', shared_with=())
    with pytest.raises(PermissionError):
        f.bindings.prepare(actor_id='member', skill_id=f.skill.skill_id, names=[name])
    assert 'REVIEWED_PRIVATE' not in f.migration.guidance(f.skill.skill_id, 'member')
    if f.kind == 'skill':
        with pytest.raises(PermissionError, match='private_legacy'):
            read_resource({'skill_id': f.skill.skill_id, 'path': 'data.txt'}, context=context(f, 'member'), store=f.skills)
    # Resume must never re-enable or broaden an existing binding.
    f.bindings.set_access(name, actor_id='owner', enabled=False)
    apply(f)
    with pytest.raises(PermissionError, match='revoked'):
        f.bindings.prepare(actor_id='owner', skill_id=f.skill.skill_id, names=[name])


def test_ambiguity_needs_reviewed_location(legacy):
    f = legacy
    other = Path(f.skill.directory) if f.kind != 'skill' else f.resources.shared_directory(f.skill.skill_id, f.record.artifact_id)
    other.mkdir(parents=True, exist_ok=True)
    (other / 'reader.py').write_text("raise RuntimeError('other copy')")
    assert preview(f)['reconciliation'] == 'ambiguous'
    with pytest.raises(ValueError, match='explicit_reconciliation'):
        apply(f)
    apply(f, location={'storage_kind': f.kind, 'entrypoint_path': 'reader.py', 'execution': f.record.execution})
    assert (other / 'reader.py').read_text() == "raise RuntimeError('other copy')"


def test_stale_preview_and_stale_location_never_guess(legacy):
    f = legacy
    token = preview(f)['review_token']
    (f.directory / 'reader.py').unlink()
    with pytest.raises(ValueError, match='preview_changed'):
        f.migration.apply('owner', f.record.artifact_id, review_token=token, instructions='Run the documented reader.')
    with pytest.raises(ValueError, match='explicit_reconciliation'):
        f.migration.apply('owner', f.record.artifact_id, review_token=preview(f)['review_token'], instructions='Run the documented reader.')
    assert (Path(f.skill.directory) / 'SKILL.md').read_bytes() == f.before


@pytest.mark.parametrize('stage', ['binding', 'instructions', 'complete'])
def test_restart_finishes_interrupted_migration(legacy, monkeypatch, stage):
    f = legacy
    token = preview(f)['review_token']
    if stage == 'binding':
        original = f.bindings.import_legacy
        def interrupted(payload):
            original(payload)
            raise OSError('interruption')
        monkeypatch.setattr(f.bindings, 'import_legacy', interrupted)
    elif stage == 'instructions':
        original = f.migration._write_instructions
        def interrupted(path, content):
            original(path, content)
            raise OSError('interruption')
        monkeypatch.setattr(f.migration, '_write_instructions', interrupted)
    else:
        monkeypatch.setattr(f.migration, '_state', lambda *args: (_ for _ in ()).throw(OSError('interruption')))
    with pytest.raises(OSError, match='interruption'):
        apply(f)
    monkeypatch.undo()
    resources = ArtifactResources(SQLiteArtifactStore(f.store.db_path), f.skills, f.projects, is_admin=f.users.is_admin)
    bindings = SkillBindings(SQLiteIntegrationStore(f.integrations.db_path), f.users, f.skills)
    restarted = LegacyMigration(resources, bindings)
    assert restarted.apply('owner', f.record.artifact_id, review_token=token, instructions='ignored')['state'] == 'complete'
    assert (f.directory / 'data.txt').read_text() == 'original household data'
    restarted.rollback('owner', f.record.artifact_id)
    assert (Path(f.skill.directory) / 'SKILL.md').read_bytes() == f.before


def test_conflicting_edit_does_not_get_overwritten_on_rollback(legacy):
    f = legacy
    apply(f)
    path = Path(f.skill.directory) / 'SKILL.md'
    path.write_text(path.read_text() + '\nUser edit\n')
    with pytest.raises(ValueError, match='rollback_conflict'):
        f.migration.rollback('owner', f.record.artifact_id)
    assert path.read_text().endswith('User edit\n')
    assert (f.directory / 'data.txt').read_text() == 'original household data'


def test_binding_preflight_rechecks_symlink_and_missing_config(legacy, tmp_path):
    f = legacy
    name = apply(f)['binding_name']
    config = f.directory / 'config.env'
    config.unlink()
    with pytest.raises(ValueError, match='prerequisite_missing'):
        f.bindings.prepare(actor_id='owner', skill_id=f.skill.skill_id, names=[name])
    outside = tmp_path / 'outside.env'
    outside.write_text('outside')
    config.symlink_to(outside)
    with pytest.raises(ValueError, match='outside_storage'):
        f.bindings.prepare(actor_id='owner', skill_id=f.skill.skill_id, names=[name])


def test_no_artifact_execution_schema_or_runtime_registration():
    from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
    assert 'artifact_id' not in BASH_ARGUMENT_SCHEMA['properties']
    tools = build_native_tool_registry(artifact_store=SQLiteArtifactStore()).list()
    assert not any('artifact' in tool.tool_id for tool in tools)


@pytest.mark.parametrize('stage', ['binding', 'instructions'])
def test_interrupted_rollback_restarts_safely(legacy, monkeypatch, stage):
    f = legacy
    apply(f)
    if stage == 'binding':
        original = f.bindings.remove_legacy
        def interrupted(payload):
            original(payload)
            raise OSError('interruption')
        monkeypatch.setattr(f.bindings, 'remove_legacy', interrupted)
    else:
        original = f.migration._write_instructions
        def interrupted(path, content):
            original(path, content)
            raise OSError('interruption')
        monkeypatch.setattr(f.migration, '_write_instructions', interrupted)
    with pytest.raises(OSError):
        f.migration.rollback('owner', f.record.artifact_id)
    monkeypatch.undo()
    restarted = LegacyMigration(f.resources, f.bindings)
    assert restarted.rollback('owner', f.record.artifact_id)['state'] == 'rolled_back'
    assert (Path(f.skill.directory) / 'SKILL.md').read_bytes() == f.before
    assert (f.directory / 'data.txt').read_text() == 'original household data'


def test_unassigned_record_needs_explicit_skill_mapping(legacy):
    f = legacy
    f.store.set_skill(f.record.artifact_id, '')
    report = preview(f)
    assert report['status'] == 'unavailable' and not report['source']['skill_id']
    result = f.migration.apply('owner', f.record.artifact_id, review_token=report['review_token'],
        instructions='Use the retained reader with the configured binding.',
        location={'skill_id': f.skill.skill_id, 'storage_kind': f.kind,
                  'entrypoint_path': f.record.entrypoint_path, 'execution': f.record.execution})
    assert result['state'] == 'complete'
    assert list(f.resources.access_records())[0].skill_id == f.skill.skill_id
    assert not f.store.get(f.record.artifact_id).skill_id  # Original provenance preserved.


def test_private_legacy_subheadings_never_load_before_migration(legacy):
    f = legacy
    ctx = context(f, 'stranger')
    text, _ = selected_skill_guidance(ctx.task, CoreLoopContext(messages=ctx.messages, skill_store=f.skills,
        artifact_resources=f.resources, skill_bindings=f.bindings))
    assert 'PRIVATE_LEGACY_SUBHEADING' not in text
    assert 'OLD_PRIVATE' not in text


def test_lifecycle_preserves_migrated_external_data_and_blocks_embedded_data(legacy, tmp_path):
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host
    f = legacy
    runtime = build_runtime_host(user_store=f.users, skill_store=f.skills, artifact_store=f.store,
        project_store=f.projects, integration_store=f.integrations)
    daemon = V2Daemon(runtime)
    with pytest.raises(ValueError, match='retained_legacy'):
        daemon.export_skill(actor_user_id='owner', skill_id=f.skill.skill_id, destination_directory=str(tmp_path / 'out'))
    apply(f)
    if f.kind == 'skill':
        with pytest.raises(ValueError, match='retained_legacy'):
            daemon.delete_skill(actor_user_id='owner', skill_id=f.skill.skill_id)
    else:
        destination = tmp_path / 'out'
        destination.mkdir()
        exported = Path(daemon.export_skill(actor_user_id='owner', skill_id=f.skill.skill_id,
            destination_directory=str(destination))['directory'])
        assert not (exported / 'data.txt').exists()
        daemon.delete_skill(actor_user_id='owner', skill_id=f.skill.skill_id)
    assert (f.directory / 'data.txt').read_text() == 'original household data'
    assert f.store.get(f.record.artifact_id) == f.record


@pytest.mark.parametrize('tool_id, arguments', [
    ('native.bash', {'artifact_id': 'artifact.fixture', 'command': 'touch unexpected'}),
    ('artifact.fixture', {}),
    ('native.skill_artifact_register', {}),
    ('native.artifact_metadata_update', {}),
])
def test_saved_references_fail_with_replanning_diagnostic(legacy, tool_id, arguments):
    from alphonse.agent_v2.core.tools.invocation import ToolInvocationService
    from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
    f = legacy
    ctx = CoreLoopContext(messages=InMemoryMessageQueue(), tools=build_native_tool_registry())
    service = ToolInvocationService(context=ctx, task=context(f).task)
    outcome = service.invoke(tool_id, arguments)
    assert outcome['status'] == 'failed'
    assert 'legacy_skill_migration' in outcome['error']['message']
    assert 'replan' in outcome['error']['message']
    assert outcome['result'] is None
    with pytest.raises(ValueError, match='execution_retired'):
        service.execute_or_raise(tool_id, arguments)
    assert not (f.directory / 'unexpected').exists()



def test_migration_preserves_virtual_environment_interpreter_path(legacy):
    import venv
    f = legacy
    environment = f.directory / '.venv'
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    executable = environment / 'bin' / 'python'
    (f.directory / 'reader.py').write_text('import sys; print(sys.prefix)')
    execution = {**f.record.execution, 'runtime_path': '.venv/bin/python'}
    f.store.update_location(replace(f.record, execution=execution))
    report = preview(f)
    assert report['runtime_path'] == str(executable)
    assert str(executable) in report['suggested_command']
    name = apply(f)['binding_name']
    outcome = execute_bash({'skill_id': f.skill.skill_id, 'binding_names': [name],
        'command': report['suggested_command']}, context=context(f))
    assert outcome['exit_code'] == 0
    assert Path(outcome['stdout'].strip()) == environment
