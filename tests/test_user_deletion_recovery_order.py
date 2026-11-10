"""Disposable regression coverage for retryable managed deletion ordering."""
from pathlib import Path

import pytest

from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def _fixture(tmp_path):
    users = V2UserStore(tmp_path / 'users.sqlite3')
    admin = users.onboard(display_name='Fixture admin', users_root=tmp_path / 'profiles')
    member = users.create_user(display_name='Fixture member')
    projects = ProjectStore(tmp_path / 'projects.sqlite3')
    root = users.managed_project_root(member.user_id) / 'project'
    project = projects.create_project(name='Fixture', root_path=str(root), owner_user_id=member.user_id)
    (root / 'note.txt').write_text('fixture', encoding='utf-8')
    daemon = V2Daemon(build_runtime_host(user=admin.user_id, user_store=users, project_store=projects))
    return users, member, projects, project, root, daemon


def test_project_cleanup_failure_preserves_owner_rows_for_retry(tmp_path, monkeypatch):
    import alphonse.agent_v2.daemon as daemon_module
    users, member, projects, project, root, daemon = _fixture(tmp_path)
    original = daemon_module.shutil.rmtree
    def fail(path, *args, **kwargs):
        if str(path) == root.name:
            raise OSError('fixture cleanup error')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(daemon_module.shutil, 'rmtree', fail)
    with pytest.raises(OSError):
        daemon.delete_user(member.user_id, confirmation=member.user_id)
    assert projects.get_project(project.project_id) is not None
    assert users.get_user(member.user_id) is not None
    assert (root / 'note.txt').read_text() == 'fixture'
    monkeypatch.setattr(daemon_module.shutil, 'rmtree', original)
    assert daemon.delete_user(member.user_id, confirmation=member.user_id)['deleted'] == 1
    assert projects.get_project(project.project_id) is None
    assert users.get_user(member.user_id) is None


def test_profile_cleanup_failure_keeps_user_and_addresses_for_retry(tmp_path, monkeypatch):
    import alphonse.agent_v2.users as users_module
    users, member, projects, project, root, daemon = _fixture(tmp_path)
    users.bind_address(user_id=member.user_id, integration_id='fixture', provider_key='fixture', provider_user_id='fixture-user')
    profile = users.users_root().resolve() / member.user_id
    original = users_module.shutil.rmtree
    def fail(path, *args, **kwargs):
        if str(path) == member.user_id:
            raise OSError('fixture profile error')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(users_module.shutil, 'rmtree', fail)
    with pytest.raises(OSError):
        daemon.delete_user(member.user_id, confirmation=member.user_id)
    assert users.get_user(member.user_id) is not None
    assert users.list_addresses(member.user_id)
    monkeypatch.setattr(users_module.shutil, 'rmtree', original)
    assert daemon.delete_user(member.user_id, confirmation=member.user_id)['deleted'] == 1


def test_project_symlink_escape_is_rejected_before_rows_or_external_data_change(tmp_path):
    users, member, projects, project, root, daemon = _fixture(tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'private.txt').write_text('fixture', encoding='utf-8')
    # Replace the recorded managed root with a symlink after registration.
    import shutil
    shutil.rmtree(root)
    root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='project_cleanup_symlink'):
        daemon.delete_user(member.user_id, confirmation=member.user_id)
    assert projects.get_project(project.project_id) is not None
    assert users.get_user(member.user_id) is not None
    assert (outside / 'private.txt').read_text() == 'fixture'


def test_profile_symlink_is_rejected_before_user_row_removal(tmp_path):
    users, member, _, _, _, _ = _fixture(tmp_path)
    profile = users.users_root() / member.user_id
    import shutil
    shutil.rmtree(profile)
    outside = tmp_path / 'outside-profile'
    outside.mkdir()
    (outside / 'private.txt').write_text('fixture', encoding='utf-8')
    profile.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='user_profile_symlink'):
        users.delete_user(member.user_id)
    assert users.get_user(member.user_id) is not None
    assert (outside / 'private.txt').read_text() == 'fixture'


def test_interrupted_project_cleanup_keeps_rows_for_idempotent_retry(tmp_path, monkeypatch):
    import alphonse.agent_v2.daemon as daemon_module
    users, member, projects, project, root, daemon = _fixture(tmp_path)
    original = daemon_module.shutil.rmtree
    def interrupted(path, *args, **kwargs):
        original(path, *args, **kwargs)
        if str(path) == root.name:
            raise KeyboardInterrupt('simulated interruption')
    monkeypatch.setattr(daemon_module.shutil, 'rmtree', interrupted)
    with pytest.raises(KeyboardInterrupt):
        daemon.delete_user(member.user_id, confirmation=member.user_id)
    assert projects.get_project(project.project_id) is not None
    assert users.get_user(member.user_id) is not None
    assert not root.exists()
    monkeypatch.setattr(daemon_module.shutil, 'rmtree', original)
    assert daemon.delete_user(member.user_id, confirmation=member.user_id)['deleted'] == 1


def test_project_record_inside_another_profile_does_not_delete_other_users_data(tmp_path):
    users, member, projects, project, root, daemon = _fixture(tmp_path)
    other = users.create_user(display_name='Other fixture member')
    protected = users.users_root() / other.user_id / 'protected'
    projects.create_project(name='Misdirected fixture', root_path=str(protected), owner_user_id=member.user_id)
    (protected / 'private.txt').write_text('fixture', encoding='utf-8')
    with pytest.raises(ValueError, match='project_cleanup_outside_owner_profile'):
        daemon.delete_user(member.user_id, confirmation=member.user_id)
    assert (root / 'note.txt').read_text() == 'fixture'
    assert projects.get_project(project.project_id) is not None
    assert users.get_user(member.user_id) is not None
    assert (protected / 'private.txt').read_text() == 'fixture'


def test_deletion_order_works_when_user_and_project_stores_share_database(tmp_path):
    db = tmp_path / 'shared.sqlite3'
    users = V2UserStore(db)
    admin = users.onboard(display_name='Fixture admin', users_root=tmp_path / 'profiles')
    member = users.create_user(display_name='Fixture member')
    projects = ProjectStore(db)
    root = users.managed_project_root(member.user_id) / 'project'
    project = projects.create_project(name='Fixture', root_path=str(root), owner_user_id=member.user_id)
    daemon = V2Daemon(build_runtime_host(user=admin.user_id, user_store=users, project_store=projects))
    assert daemon.delete_user(member.user_id, confirmation=member.user_id)['deleted'] == 1
    assert projects.get_project(project.project_id) is None
    assert users.get_user(member.user_id) is None
    assert not root.exists()


def test_interrupted_profile_cleanup_keeps_user_row_for_idempotent_retry(tmp_path, monkeypatch):
    import alphonse.agent_v2.users as users_module
    users, member, _, _, _, _ = _fixture(tmp_path)
    profile = users.users_root().resolve() / member.user_id
    original = users_module.shutil.rmtree
    def interrupted(path, *args, **kwargs):
        original(path, *args, **kwargs)
        if str(path) == member.user_id:
            raise KeyboardInterrupt('simulated interruption')
    monkeypatch.setattr(users_module.shutil, 'rmtree', interrupted)
    with pytest.raises(KeyboardInterrupt):
        users.delete_user(member.user_id)
    assert users.get_user(member.user_id) is not None
    assert not profile.exists()
    monkeypatch.setattr(users_module.shutil, 'rmtree', original)
    assert users.delete_user(member.user_id) is True
    assert users.get_user(member.user_id) is None


def test_parent_symlink_swap_during_project_removal_never_deletes_external_data(tmp_path, monkeypatch):
    import alphonse.agent_v2.workspace as workspace_module
    users, member, projects, project, root, daemon = _fixture(tmp_path)
    profile = users.users_root().resolve() / member.user_id
    moved = profile.with_name(profile.name + '-moved')
    outside = tmp_path / 'outside-race'
    (outside / root.name).mkdir(parents=True)
    protected = outside / root.name / 'private.txt'
    protected.write_text('fixture', encoding='utf-8')
    original = workspace_module.shutil.rmtree
    swapped = False
    def swap_parent(path, *args, **kwargs):
        nonlocal swapped
        if str(path) == root.name and not swapped:
            swapped = True
            profile.rename(moved)
            profile.symlink_to(outside, target_is_directory=True)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(workspace_module.shutil, 'rmtree', swap_parent)
    # The later profile check notices the replacement; the already-open parent
    # descriptor must never follow it into the external directory.
    with pytest.raises(ValueError, match='user_profile_symlink'):
        daemon.delete_user(member.user_id, confirmation=member.user_id)
    assert swapped
    assert protected.read_text() == 'fixture'
    assert users.get_user(member.user_id) is not None


def test_missing_inner_entry_does_not_mark_project_cleanup_complete(tmp_path, monkeypatch):
    import alphonse.agent_v2.workspace as workspace_module
    users, member, projects, project, root, daemon = _fixture(tmp_path)
    original = workspace_module.shutil.rmtree
    def missing_child(path, *args, **kwargs):
        if str(path) == root.name:
            raise FileNotFoundError('simulated disappearing child')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(workspace_module.shutil, 'rmtree', missing_child)
    with pytest.raises(FileNotFoundError):
        daemon.delete_user(member.user_id, confirmation=member.user_id)
    assert root.exists()
    assert projects.get_project(project.project_id) is not None
    assert users.get_user(member.user_id) is not None


def test_missing_users_root_is_not_treated_as_successful_profile_cleanup(tmp_path):
    users, member, _, _, _, _ = _fixture(tmp_path)
    root = users.users_root()
    moved = root.with_name('profiles-unavailable')
    root.rename(moved)
    assert (moved / member.user_id).exists()
    with pytest.raises(ValueError, match='managed_cleanup_root_unavailable'):
        users.delete_user(member.user_id)
    assert users.get_user(member.user_id) is not None
    assert (moved / member.user_id).exists()


def test_platform_without_safe_fd_removal_fails_closed(tmp_path, monkeypatch):
    import alphonse.agent_v2.workspace as workspace_module
    users, member, projects, project, root, daemon = _fixture(tmp_path)
    monkeypatch.setattr(workspace_module, '_SAFE_RMTREE_SUPPORTED', False)
    with pytest.raises(ValueError, match='managed_cleanup_unsupported'):
        daemon.delete_user(member.user_id, confirmation=member.user_id)
    assert (root / 'note.txt').read_text() == 'fixture'
    assert projects.get_project(project.project_id) is not None
    assert users.get_user(member.user_id) is not None
