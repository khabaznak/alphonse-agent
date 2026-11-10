from types import SimpleNamespace

import pytest

import alphonse.agent_v2.coding_skill_runtime as module
from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.users import V2UserStore


def test_factory_requires_explicit_immutable_image_and_persistent_household(tmp_path, monkeypatch):
    monkeypatch.delenv('ALPHONSE_CODEX_SANDBOX_IMAGE_ID', raising=False)
    home = tmp_path / 'codex-home'
    home.mkdir()
    (home / 'auth.json').write_text('{}')
    monkeypatch.setenv('CODEX_HOME', str(home))
    monkeypatch.setattr(module.shutil, 'which', lambda _: '/bin/echo')
    workspace = tmp_path / 'household'
    workspace.mkdir()
    users = SimpleNamespace(is_ephemeral=False, workspace_root=lambda: workspace,
                            db_path=str(tmp_path / 'database.sqlite'), is_admin=lambda _: True)
    skills = SkillStore(workspace / 'skills')
    assert module.build_configured_coding_skill_service(users=users, skills=skills) is None
    monkeypatch.setenv('ALPHONSE_CODEX_SANDBOX_IMAGE_ID', 'sha256:' + 'a' * 64)
    service = module.build_configured_coding_skill_service(users=users, skills=skills)
    assert service is not None and service.skills is skills
    assert service.provider.runner is service.verifier.runner
    users.is_ephemeral = True
    with pytest.raises(ValueError, match='persistent_workspace_required'):
        module.build_configured_coding_skill_service(users=users, skills=skills)


def test_factory_rejects_mutable_image_and_symlinked_root(tmp_path, monkeypatch):
    home = tmp_path / 'codex-home'
    home.mkdir()
    (home / 'auth.json').write_text('{}')
    monkeypatch.setenv('CODEX_HOME', str(home))
    monkeypatch.setattr(module.shutil, 'which', lambda _: '/bin/echo')
    workspace = tmp_path / 'household'
    workspace.mkdir()
    users = SimpleNamespace(is_ephemeral=False, workspace_root=lambda: workspace,
                            db_path=str(tmp_path / 'database.sqlite'), is_admin=lambda _: True)
    skills = SkillStore(workspace / 'skills')
    monkeypatch.setenv('ALPHONSE_CODEX_SANDBOX_IMAGE_ID', 'mutable-tag')
    with pytest.raises(ValueError, match='configuration_invalid'):
        module.build_configured_coding_skill_service(users=users, skills=skills)
    (workspace / '.coding-jobs').rmdir()
    (workspace / '.coding-jobs').symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setenv('ALPHONSE_CODEX_SANDBOX_IMAGE_ID', 'sha256:' + 'a' * 64)
    with pytest.raises(ValueError, match='workspace_root_invalid'):
        module.build_configured_coding_skill_service(users=users, skills=skills)


def test_persistent_image_setting_is_validated():
    users = V2UserStore()
    image = 'sha256:' + 'a' * 64
    assert users.set_coding_sandbox_image_id(image) == image
    assert users.coding_sandbox_image_id() == image
    with pytest.raises(ValueError, match='image_id_invalid'):
        users.set_coding_sandbox_image_id('mutable-tag')
    assert users.coding_sandbox_image_id() == image
    users.set_coding_sandbox_image_id('')
    assert users.coding_sandbox_image_id() == ''


def test_misconfigured_optional_delegation_does_not_block_runtime(monkeypatch):
    from alphonse.agent_v2.runtime import build_runtime_host
    monkeypatch.setenv('ALPHONSE_CODEX_SANDBOX_IMAGE_ID', 'sha256:' + 'a' * 64)
    runtime = build_runtime_host()
    assert runtime.coding_skill_service is None
    assert runtime.coding_skill_error == 'coding_skill_persistent_workspace_required'
    assert runtime.core.tools.get('native.coding_skill_build') is None
