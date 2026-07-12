from pathlib import Path
import pytest

from alphonse.agent_v2.core.io import IntegrationIdentity, V2IdentityResolver
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def test_onboarding_creates_uuid_admin_profile_and_context(tmp_path: Path) -> None:
    store = V2UserStore(tmp_path / "users.sqlite3")
    admin = store.onboard(display_name="Alex", users_root=tmp_path / "profiles")

    assert admin.role == "admin"
    assert admin.user_id not in {"local", "admin"}
    assert (tmp_path / "profiles" / admin.user_id / "user_context.md").exists()
    assert store.status()["onboarded"] is True


def test_v2_address_mapping_resolves_without_v1_identity(tmp_path: Path) -> None:
    store = V2UserStore(tmp_path / "users.sqlite3")
    alex = store.onboard(display_name="Alex", users_root=tmp_path / "profiles")
    store.bind_address(user_id=alex.user_id, integration_id="telegram-home", provider_key="telegram", provider_user_id="123", channel_target="456")
    resolver = V2IdentityResolver((IntegrationIdentity("telegram-home", "telegram"),), user_store=store)

    inbound = resolver.resolve_inbound_user(integration_id="telegram-home", provider_key="telegram", provider_user_id="123")
    outbound = resolver.resolve_outbound_address(alphonse_user_id=alex.user_id)

    assert inbound.alphonse_user_id == alex.user_id
    assert outbound.address is not None and outbound.address.channel_target == "456"


def test_non_admin_profile_projects_and_membership_are_isolated(tmp_path: Path) -> None:
    store = V2UserStore(tmp_path / "users.sqlite3")
    admin = store.onboard(display_name="Alex", users_root=tmp_path / "profiles")
    gaby = store.create_user(display_name="Gaby")
    projects = ProjectStore(tmp_path / "projects.sqlite3")
    project = projects.create_project(name="Health", root_path=str(store.managed_project_root(gaby.user_id) / "health"), owner_user_id=gaby.user_id)

    assert Path(project.context_path).exists()
    assert projects.get_project(project.project_id, requester_user_id=admin.user_id) is None
    projects.add_member(project.project_id, admin.user_id)
    assert projects.get_project(project.project_id, requester_user_id=admin.user_id) is not None
    with pytest.raises(PermissionError):
        projects.write_project_context(project.project_id, "cannot edit", requester_user_id=admin.user_id)


def test_onboarding_migrates_legacy_local_projects(tmp_path: Path) -> None:
    users = V2UserStore(tmp_path / "users.sqlite3")
    projects = ProjectStore(tmp_path / "projects.sqlite3")
    legacy = projects.create_project(name="Legacy", root_path=str(tmp_path / "legacy"), owner_user_id="local")
    runtime = build_runtime_host(user_store=users, project_store=projects)
    daemon = V2Daemon(runtime)

    result = daemon.onboard(display_name="Alex", users_root=str(tmp_path / "profiles"))

    assert result["migration"]["local_projects_migrated"] == 1
    assert projects.get_project(legacy.project_id).owner_user_id == result["admin_user"]["user_id"]
