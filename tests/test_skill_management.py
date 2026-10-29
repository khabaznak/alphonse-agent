"""Package management uses disposable resources and preserves household state."""
from pathlib import Path

import pytest

from alphonse.agent_v2.skills import SkillStore


def package(parent, name='sample', body='Use the fixture.'):
    root = parent / name
    root.mkdir(parents=True)
    (root / 'SKILL.md').write_text(f'---\nname: {name}\ndescription: A fixture\nlicense: MIT\nmetadata:\n  author: fixture\n---\n{body}\n')
    (root / 'assets/nested').mkdir(parents=True)
    (root / 'assets/nested/image.bin').write_bytes(b'\x00\xffimage')
    return root


@pytest.fixture
def store(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / 'skills')
    bundled = tmp_path / 'bundled'
    package(bundled)
    monkeypatch.setattr(store, '_bundled_skills_dir', lambda: bundled)
    return store


def test_customize_conflict_update_export_uninstall(store, tmp_path):
    original = store.get('sample')
    assert store.management_status('sample')['kind'] == 'bundled'
    customized = store.customize('sample', owner='alex')
    assert customized.directory != original.directory
    assert store.management_status('sample')['owner'] == 'alex'
    store.write_file('sample', 'references/nested/household.md', 'Local instructions')
    assert store.management_status('sample')['customized']
    incoming = package(tmp_path / 'upstream', body='New upstream instructions')
    with pytest.raises(ValueError, match='skill_update_conflict'):
        store.replace_directory('sample', incoming)
    assert store.read_file('sample', 'references/nested/household.md') == 'Local instructions'
    store.replace_directory('sample', incoming, resolution='accept-upstream')
    recovery = store.skills_dir / '.skill-state/sample'
    assert list(recovery.glob('*/sample/references/nested/household.md'))
    destination = tmp_path / 'export'
    destination.mkdir()
    exported = store.export_directory('sample', destination)
    assert (exported / 'assets/nested/image.bin').read_bytes() == b'\x00\xffimage'
    assert not (exported / 'provenance.json').exists()
    # Uninstall retains legacy colocated data even though new bindings must be external.
    (Path(customized.directory) / 'legacy.sqlite').write_bytes(b'private data')
    store.uninstall('sample')
    assert list(recovery.glob('*/sample/legacy.sqlite'))[0].read_bytes() == b'private data'
    assert store.get('sample').directory == original.directory
    assert Path(original.directory, 'SKILL.md').read_text().endswith('Use the fixture.\n')


def test_binary_metadata_and_path_authority(store, tmp_path):
    store.customize('sample')
    result = store.inspect_file('sample', 'assets/nested/image.bin')
    assert result['content'] is None and not result['text_editable']
    with pytest.raises(ValueError, match='not_text_editable'):
        store.write_file('sample', 'assets/nested/image.bin', 'do not overwrite binary content')
    definition = store.inspect_file('sample', 'SKILL.md')
    assert definition['text_editable']
    store.write_file('sample', 'SKILL.md', definition['content'].replace('license: MIT', 'license: Apache-2.0'))
    assert store.get('sample').license == 'Apache-2.0'
    before = store.read_file('sample', 'SKILL.md')
    with pytest.raises(ValueError, match='description_invalid'):
        store.write_file('sample', 'SKILL.md', before.replace('description: A fixture', 'description: false'))
    assert store.read_file('sample', 'SKILL.md') == before
    for path in ('./SKILL.md', 'assets/../SKILL.md'):
        with pytest.raises(ValueError, match='definition_required'):
            store.delete_file('sample', path)
    for path in ('../outside', '/tmp/outside'):
        with pytest.raises(ValueError):
            store.write_file('sample', path, 'denied')
    root = Path(store.get('sample').directory)
    (root / 'link').symlink_to(root / 'assets', target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        store.write_file('sample', 'link/file.txt', 'denied')


def test_export_refuses_conventional_private_state(store, tmp_path):
    store.customize('sample')
    root = Path(store.get('sample').directory)
    (root / '.env').write_text('PRIVATE=value')
    with pytest.raises(ValueError, match='export_private_state'):
        store.export_directory('sample', tmp_path)
    with pytest.raises(ValueError, match='update_private_state'):
        store.replace_directory('sample', store._bundled_skills_dir() / 'sample', resolution='accept-upstream')
    assert (root / '.env').read_text() == 'PRIVATE=value'


def test_ipc_management_authorization_and_readiness(store, tmp_path):
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host
    from alphonse.agent_v2.users import V2UserStore
    users = V2UserStore(':memory:')
    users.set_workspace_root(tmp_path)
    users.onboard(display_name='Alex', user_id='alex')
    daemon = V2Daemon(build_runtime_host(user_store=users, skill_store=store))
    dispatch = daemon.ipc._dispatch
    def request(method, **params):
        return dispatch({'method': method, 'params': {'actor_user_id': 'alex', 'skill_id': 'skill:sample', **params}})
    for method in ('skills', 'customize_skill', 'read_skill_file', 'write_skill_file', 'install_skill', 'replace_skill', 'export_skill', 'delete_skill', 'add_skill_file', 'delete_skill_file', 'update_skill'):
        with pytest.raises(PermissionError):
            request(method, actor_user_id='stranger')
    users.create_user(display_name='Member', user_id='member')
    with pytest.raises(PermissionError):
        request('customize_skill', actor_user_id='member')
    source = package(tmp_path / 'source', name='installed')
    installed = request('install_skill', source_directory=str(source))['skill']
    assert installed['id'] == 'skill:installed'
    request('replace_skill', skill_id='skill:installed', source_directory=str(source))
    with pytest.raises(ValueError, match='bundled_name_conflict'):
        request('install_skill', source_directory=store.get('sample').directory)
    request('customize_skill')
    result = request('read_skill_file', path='assets/nested/image.bin')
    assert result['content'] is None and not result['text_editable']
    bindings = daemon.runtime.core.skill_bindings
    bindings.configure(actor_id='alex', binding_id='fixture', skill_id='skill:sample', owner_id='alex', kind='configuration', configuration={}, executables=('nonexistent-fixture-executable',))
    bindings.select(actor_id='alex', skill_id='skill:sample', name='fixture', binding_id='fixture')
    status = next(skill for skill in request('skills')['skills'] if skill['id'] == 'skill:sample')
    assert status['bindings'] == [{'name': 'fixture', 'status': 'skill_binding_dependency_missing'}]
    request('delete_skill')
    assert bindings._get('fixture')['enabled']


def test_invalid_update_keeps_active_revision_and_bundled_updates_are_visible(store, tmp_path):
    store.customize('sample')
    original = store.management_status('sample')['local_revision']
    bundled = store._bundled_skills_dir() / 'sample/SKILL.md'
    bundled.write_text(bundled.read_text() + '\nUpdated bundled instructions')
    assert store.management_status('sample')['bundled_update_available']
    source = package(tmp_path / 'incoming')
    (source / 'SKILL.md').write_text('invalid')
    with pytest.raises(ValueError):
        store.replace_directory('sample', source)
    assert store.management_status('sample')['local_revision'] == original


def test_harness_state_cannot_escape_by_symlink(store, tmp_path):
    (store.skills_dir / '.skill-state').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='state_symlink'):
        store.customize('sample')
    assert not (store.skills_dir / 'sample').exists()


def test_large_asset_import_and_legacy_link_diagnostics(store, tmp_path):
    store.customize('sample')
    source = tmp_path / 'large.bin'
    source.write_bytes(b'\x00' * 300_000)
    store.add_file_from_path('sample', source)
    assert store.inspect_file('sample', 'large.bin')['content'] is None
    root = Path(store.get('sample').directory)
    (root / 'legacy-environment').symlink_to(tmp_path)
    assert store.list_skills()[0].name == 'sample'
    assert 'symlink' in store.management_status('sample')['validation_error']
