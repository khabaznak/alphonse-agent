"""Agent Skills conformance and Alphonse package lifecycle regressions."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import jsonschema
import pytest
import yaml

# Initialize intelligence before native tools (the native package has a legacy import cycle).
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.skills import SkillStore, MAX_SKILL_FILE_BYTES
from alphonse.agent_v2.core.tools.registry.native.skill_install import (
    SKILL_INSTALL_ARGUMENT_SCHEMA, execute_skill_install,
)

FIXTURES = Path(__file__).parent / 'fixtures' / 'skills'
CASES = json.loads((FIXTURES / 'validation.json').read_text())


def package(parent, metadata=None, body='Read references/guide.md when needed.', name='sample'):
    root = parent / name
    root.mkdir(parents=True)
    (root / 'SKILL.md').write_text('---\n' + yaml.safe_dump(metadata if metadata is not None else {'name': name, 'description': 'A fixture'}) + '---\n' + body)
    return root


@pytest.fixture
def store(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / 'installed')
    monkeypatch.setattr(store, '_bundled_skills_dir', lambda: tmp_path / 'no-bundled')
    return store


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['id'])
def test_package_validation_fixtures(store, tmp_path, case):
    root = package(tmp_path / 'source', case['frontmatter'], name=case.get('directory', 'sample'), body=case.get('body', 'Use this fixture.'))
    if case['valid']:
        record = store.install_directory(root)
        assert record.description == case['frontmatter']['description']
    else:
        with pytest.raises(ValueError, match=case['error']):
            store.install_directory(root)


@pytest.mark.parametrize('case', [case for case in CASES if case.get('reference', True)], ids=lambda case: case['id'])
def test_reference_validator_agrees_on_shared_rules(tmp_path, case):
    validator = pytest.importorskip('skills_ref.validator', reason='optional official skills-ref checkout required')
    root = package(tmp_path, case['frontmatter'], name=case.get('directory', 'sample'), body=case.get('body', 'Use this fixture.'))
    assert (not validator.validate_metadata(case['frontmatter'], root)) == case['valid']


@pytest.mark.parametrize('length', [512, 513, 1024, 1025])
def test_description_boundary_across_authoring_update_and_schema(store, length):
    description = 'd' * length
    arguments = {'name': 'sample', 'description': description, 'instructions': 'Use the fixture.'}
    context = SimpleNamespace(task=SimpleNamespace(user='admin'))
    if length > 1024:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(arguments, SKILL_INSTALL_ARGUMENT_SCHEMA)
        with pytest.raises(ValueError, match='skill_description_invalid'):
            store.create_skill(**arguments)
        return
    jsonschema.validate(arguments, SKILL_INSTALL_ARGUMENT_SCHEMA)
    result = execute_skill_install(arguments, context=context, store=store, is_admin=lambda actor: actor == 'admin')
    assert result['skill']['description'] == description
    assert store.update_description('sample', description).description == description
    with pytest.raises(ValueError, match='skill_description_invalid'):
        store.update_description('sample', 'x' * 1025)


@pytest.mark.parametrize('description', [True, 42, ['text'], {'text': 'value'}])
def test_authoring_rejects_wrong_description_types(store, description):
    with pytest.raises(ValueError, match='skill_description_invalid'):
        execute_skill_install({'name': 'sample', 'description': description, 'instructions': 'Use fixture.'},
                              context=SimpleNamespace(task=SimpleNamespace(user='admin')), store=store, is_admin=lambda actor: True)


def snapshot(root):
    return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mode & 0o777) for p in root.rglob('*') if p.is_file()}


def test_install_replace_export_preserve_entire_package(store, tmp_path):
    source = tmp_path / 'source' / 'sample'
    shutil.copytree(FIXTURES / 'sample', source)
    (source / 'scripts' / 'nested' / 'read.sh').chmod(0o755)
    original = snapshot(source)
    record = store.install_directory(source)
    assert snapshot(Path(record.directory)) == original
    assert record.package_metadata() == {'license': 'LICENSE.txt', 'compatibility': 'Requires a shell', 'metadata': {'author': 'Alphonse test', 'version': '1.0'}, 'allowed-tools': 'Bash Read'}
    updated = store.update_description(record.skill_id, 'Updated description')
    assert updated.package_metadata() == record.package_metadata()
    assert store.get(record.skill_id).allowed_tools == 'Bash Read'
    (source / 'assets' / 'nested' / 'fixture.bin').write_bytes(b'\x00\xffreplacement')
    with pytest.raises(ValueError, match='skill_update_conflict'):
        store.replace_directory(record.skill_id, source)
    store.replace_directory(record.skill_id, source, resolution='accept-upstream')
    destination = tmp_path / 'export'
    destination.mkdir()
    exported = store.export_directory(record.skill_id, destination)
    assert snapshot(exported) == snapshot(source)
    with pytest.raises(ValueError, match='destination_exists'):
        store.export_directory(record.skill_id, destination)
    assert snapshot(source) == snapshot(exported)


@pytest.mark.parametrize('operation', ['install', 'replace', 'export'])
def test_package_copy_rejects_symlink_escape(store, tmp_path, operation):
    source = package(tmp_path / 'source')
    record = store.install_directory(source) if operation != 'install' else None
    root = Path(record.directory) if operation == 'export' else source
    (root / 'outside').symlink_to(tmp_path)
    with pytest.raises(ValueError, match='skill_operational_symlink_not_allowed'):
        if operation == 'install':
            store.install_directory(source)
        elif operation == 'replace':
            store.replace_directory(record.skill_id, source)
        else:
            store.export_directory(record.skill_id, tmp_path)


def test_listing_exposes_specification_operational_and_filesystem_diagnostics(store):
    package(store.skills_dir, {'name': 'invalid', 'description': True}, name='invalid')
    oversized = package(store.skills_dir, name='oversized')
    (oversized / 'SKILL.md').write_text('x' * (MAX_SKILL_FILE_BYTES + 1))
    (store.skills_dir / 'missing').mkdir()
    package(store.skills_dir, name='valid')
    assert [s.name for s in store.list_skills()] == ['valid']
    assert {d['category'] for d in store.diagnostics} == {'specification', 'operational', 'filesystem'}
    assert all(d['directory'] and d['error'] for d in store.diagnostics)


def test_ipc_metadata_diagnostics_export_and_authorization(store, tmp_path):
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host
    from alphonse.agent_v2.users import V2UserStore
    source = tmp_path / 'source' / 'sample'
    shutil.copytree(FIXTURES / 'sample', source)
    store.install_directory(source)
    package(store.skills_dir, {'name': 'invalid', 'description': False}, name='invalid')
    users = V2UserStore(':memory:')
    users.set_workspace_root(tmp_path)
    users.onboard(display_name='Alex', user_id='alex')
    daemon = V2Daemon(build_runtime_host(user_store=users, skill_store=store))
    result = daemon.ipc._dispatch({'method': 'skills', 'params': {'actor_user_id': 'alex'}})
    assert result['skills'][0]['metadata']['version'] == '1.0'
    assert result['skills'][0]['allowed-tools'] == 'Bash Read'
    assert result['diagnostics'][0]['category'] == 'specification'
    destination = tmp_path / 'export'
    destination.mkdir()
    params = {'actor_user_id': 'alex', 'skill_id': 'sample', 'destination_directory': str(destination)}
    exported = daemon.ipc._dispatch({'method': 'export_skill', 'params': params})
    assert (Path(exported['directory']) / 'scripts/nested/read.sh').is_file()
    with pytest.raises(PermissionError):
        daemon.ipc._dispatch({'method': 'export_skill', 'params': {**params, 'actor_user_id': 'stranger'}})


def test_metadata_numeric_key_is_rejected(store, tmp_path):
    root = package(tmp_path / 'source', {'name': 'sample', 'description': 'A fixture', 'metadata': {1: 'value'}})
    with pytest.raises(ValueError, match='skill_metadata_invalid'):
        store.install_directory(root)


def test_reference_validator_accepts_complete_package():
    validator = pytest.importorskip('skills_ref.validator')
    assert validator.validate(FIXTURES / 'sample') == []


@pytest.mark.parametrize('content', ['name: sample', '---\nname: sample', '---\n- value\n---', '---\nname: [\n---'])
def test_malformed_frontmatter_is_diagnosed(store, tmp_path, content):
    root = store.skills_dir / 'sample'
    root.mkdir()
    (root / 'SKILL.md').write_text(content)
    assert store.list_skills() == []
    assert store.diagnostics[0]['category'] == 'specification'


def test_import_body_has_no_authoring_length_constraint(store, tmp_path):
    root = package(tmp_path / 'source', body='x' * 13000)
    assert len(store.install_directory(root).instructions) == 13000
    with pytest.raises(ValueError, match='skill_operational_instructions_too_long'):
        store.create_skill('authored', 'A fixture', 'x' * 13000)


def test_description_limits_count_unicode_characters(store):
    assert store.create_skill('unicode', '😀' * 1024, 'Use fixture.').description == '😀' * 1024
    with pytest.raises(ValueError, match='skill_description_invalid'):
        store.update_description('unicode', '😀' * 1025)
