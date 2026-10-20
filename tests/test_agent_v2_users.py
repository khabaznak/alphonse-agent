from pathlib import Path
import pytest

from alphonse.agent_v2.core.io import IntegrationIdentity, V2IdentityResolver
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def test_user_timezone_persists_when_store_is_reopened(tmp_path) -> None:
    database = tmp_path / "users.sqlite3"
    store = V2UserStore(database)
    store.set_timezone("America/Mexico_City")

    reopened = V2UserStore(database)

    assert reopened.timezone() == "America/Mexico_City"


def test_timezone_can_be_configured_per_user_with_global_fallback() -> None:
    store = V2UserStore(":memory:")
    store.set_timezone("UTC")
    store.set_timezone("America/Mexico_City", user_id="member-a")

    assert store.timezone("member-a") == "America/Mexico_City"
    assert store.timezone("member-b") == "UTC"


def test_automation_preferred_channel_copy_setting_defaults_off_and_persists(tmp_path) -> None:
    database = tmp_path / "users.sqlite3"
    store = V2UserStore(database)

    assert store.mirror_automation_messages_to_preferred_channel() is False
    assert store.set_mirror_automation_messages_to_preferred_channel(True) is True

    reopened = V2UserStore(database)
    assert reopened.status()["mirror_automation_messages_to_preferred_channel"] is True


def test_onboarding_creates_uuid_admin_profile_and_context(tmp_path: Path) -> None:
    store = V2UserStore(tmp_path / "users.sqlite3")
    admin = store.onboard(display_name="Alex", users_root=tmp_path / "profiles")

    assert admin.role == "admin"
    assert admin.user_id not in {"local", "admin"}
    assert (tmp_path / "profiles" / admin.user_id / "user_context.md").exists()
    assert store.status()["onboarded"] is True


def test_workspace_root_is_required_state_and_migrates_user_data_idempotently(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy-users"
    store = V2UserStore(":memory:")
    admin = store.onboard(display_name="Alex", users_root=legacy, user_id="alex")
    (legacy / admin.user_id / "user_context.md").write_text("Family context", encoding="utf-8")
    (legacy / admin.user_id / "memory" / "daily" / "2026-10-13.jsonl").parent.mkdir(parents=True)
    (legacy / admin.user_id / "memory" / "daily" / "2026-10-13.jsonl").write_text("ledger", encoding="utf-8")

    assert store.status()["workspace_configured"] is False
    workspace = tmp_path / "Workspace"
    store.set_workspace_root(workspace)
    store.set_workspace_root(workspace)
    second_workspace = tmp_path / "Workspace 2"
    store.set_workspace_root(second_workspace)

    assert store.workspace_root() == second_workspace.resolve()
    assert store.users_root() == second_workspace / "users"
    assert store.skills_root() == second_workspace / "skills"
    assert store.agent_config_root() == second_workspace / "agent-config"
    assert (second_workspace / "users" / admin.user_id / "user_context.md").read_text(encoding="utf-8") == "Family context"
    assert (second_workspace / "users" / admin.user_id / "memory" / "daily" / "2026-10-13.jsonl").read_text(encoding="utf-8") == "ledger"
    assert (legacy / admin.user_id / "user_context.md").exists()


def test_workspace_migration_moves_managed_projects_and_keeps_imported_roots(tmp_path: Path) -> None:
    users = V2UserStore(":memory:")
    admin = users.onboard(display_name="Alex", users_root=tmp_path / "legacy-users", user_id="alex")
    projects = ProjectStore(tmp_path / "projects.sqlite3")
    managed = projects.create_project(
        name="Managed", root_path=str(users.managed_project_root(admin.user_id) / "managed"), owner_user_id=admin.user_id,
    )
    (Path(managed.root_path) / "notes.txt").write_text("keep", encoding="utf-8")
    imported_root = tmp_path / "external" / "Imported"
    imported = projects.create_project(name="Imported", root_path=str(imported_root), owner_user_id=admin.user_id)
    old_managed_root = users.managed_project_root(admin.user_id)
    workspace = tmp_path / "Workspace"

    for _ in range(2):
        projects.migrate_managed_roots({admin.user_id: old_managed_root}, workspace)

    migrated = projects.get_project(managed.project_id)
    untouched = projects.get_project(imported.project_id)
    assert migrated is not None and Path(migrated.root_path) == workspace / "projects" / admin.user_id / managed.project_id
    assert (Path(migrated.root_path) / "notes.txt").read_text(encoding="utf-8") == "keep"
    assert untouched is not None and Path(untouched.root_path) == imported_root.resolve()


def test_v2_address_mapping_resolves_without_v1_identity(tmp_path: Path) -> None:
    store = V2UserStore(tmp_path / "users.sqlite3")
    alex = store.onboard(display_name="Alex", users_root=tmp_path / "profiles")
    store.bind_address(user_id=alex.user_id, integration_id="telegram-home", provider_key="telegram", provider_user_id="123", channel_target="456")
    resolver = V2IdentityResolver((IntegrationIdentity("telegram-home", "telegram"),), user_store=store)

    inbound = resolver.resolve_inbound_user(integration_id="telegram-home", provider_key="telegram", provider_user_id="123")
    outbound = resolver.resolve_outbound_address(alphonse_user_id=alex.user_id)

    assert inbound.alphonse_user_id == alex.user_id
    assert outbound.address is not None and outbound.address.channel_target == "456"


def test_user_aliases_are_unique_and_limited_to_two(tmp_path: Path) -> None:
    store = V2UserStore(tmp_path / "users.sqlite3")
    alex = store.onboard(display_name="Alex", users_root=tmp_path / "profiles")
    gaby = store.create_user(display_name="Gabriela Villasana Rodríguez")

    assert store.set_aliases(gaby.user_id, ["Gaby", "Mamá"]) == ["Gaby", "Mamá"]
    assert store.resolve_user_reference("gabriela villasana rodriguez")[0] == "resolved"
    assert store.resolve_user_reference("Gaby")[1][0].user_id == gaby.user_id
    with pytest.raises(ValueError, match="user_alias_already_assigned"):
        store.set_aliases(alex.user_id, ["Gaby"])
    with pytest.raises(ValueError, match="user_alias_limit_exceeded"):
        store.set_aliases(gaby.user_id, ["Gaby", "Mamá", "Gab"])


def test_non_admin_profile_projects_and_membership_are_isolated(tmp_path: Path) -> None:
    store = V2UserStore(tmp_path / "users.sqlite3")
    admin = store.onboard(display_name="Alex", users_root=tmp_path / "profiles")
    gaby = store.create_user(display_name="Gaby")
    projects = ProjectStore(tmp_path / "projects.sqlite3")
    project = projects.create_project(name="Health", root_path=str(store.managed_project_root(gaby.user_id) / "health"), owner_user_id=gaby.user_id)

    assert Path(project.config_path).exists()
    assert projects.get_project(project.project_id, requester_user_id=admin.user_id) is None
    projects.update_project(project.project_id, name=project.name, description="", visibility="shared", requester_user_id=gaby.user_id)
    projects.add_member(project.project_id, admin.user_id)
    assert projects.get_project(project.project_id, requester_user_id=admin.user_id) is not None
    with pytest.raises(PermissionError):
        projects.write_project_config(project.project_id, {"scope": "cannot edit"}, requester_user_id=admin.user_id)


def test_onboarding_migrates_legacy_local_projects(tmp_path: Path) -> None:
    users = V2UserStore(tmp_path / "users.sqlite3")
    projects = ProjectStore(tmp_path / "projects.sqlite3")
    legacy = projects.create_project(name="Legacy", root_path=str(tmp_path / "legacy"), owner_user_id="local")
    runtime = build_runtime_host(user_store=users, project_store=projects)
    daemon = V2Daemon(runtime)

    result = daemon.onboard(display_name="Alex", users_root=str(tmp_path / "profiles"))

    assert result["migration"]["local_projects_migrated"] == 1
    assert projects.get_project(legacy.project_id).owner_user_id == result["admin_user"]["user_id"]


def test_hard_delete_removes_user_profile_projects_and_addresses(tmp_path: Path) -> None:
    users = V2UserStore(tmp_path / "users.sqlite3")
    admin = users.onboard(display_name="Alex", users_root=tmp_path / "profiles")
    gaby = users.create_user(display_name="Gaby")
    projects = ProjectStore(tmp_path / "projects.sqlite3")
    project = projects.create_project(name="Health", root_path=str(users.managed_project_root(gaby.user_id) / "health"), owner_user_id=gaby.user_id)
    users.bind_address(user_id=gaby.user_id, integration_id="telegram-home", provider_key="telegram", provider_user_id="123")
    daemon = V2Daemon(build_runtime_host(user=admin.user_id, user_store=users, project_store=projects))

    result = daemon.delete_user(gaby.user_id, confirmation=gaby.user_id)

    assert result["deleted"] == 1
    assert projects.get_project(project.project_id) is None
    assert users.get_user(gaby.user_id) is None
    assert not (tmp_path / "profiles" / gaby.user_id).exists()


def test_daemon_approval_binds_request_and_allows_its_private_chat(tmp_path: Path) -> None:
    users = V2UserStore(tmp_path / "users.sqlite3")
    admin = users.onboard(display_name="Alex", users_root=tmp_path / "profiles")
    integrations = SQLiteIntegrationStore(tmp_path / "integrations.sqlite3")
    integrations.upsert(
        integration_id="telegram-home", provider_key="telegram", display_name="Telegram",
        enabled=True, config={"allowed_chat_ids": ["111"]}, secrets={},
    )
    request = users.record_access_request(
        integration_id="telegram-home", provider_key="telegram", provider_user_id="222",
        channel_target="222", display_name="Gaby",
    )
    daemon = V2Daemon(build_runtime_host(user=admin.user_id, user_store=users, integration_store=integrations))

    result = daemon.approve_access_request(request_id=request.request_id)

    assert result["address"]["integration_id"] == "telegram-home"
    assert "222" in integrations.get("telegram-home").config["allowed_chat_ids"]


def test_hard_delete_requires_exact_confirmation_and_never_deletes_admin(tmp_path: Path) -> None:
    users = V2UserStore(tmp_path / "users.sqlite3")
    admin = users.onboard(display_name="Alex", users_root=tmp_path / "profiles")
    daemon = V2Daemon(build_runtime_host(user=admin.user_id, user_store=users))

    with pytest.raises(ValueError):
        daemon.delete_user(admin.user_id, confirmation=admin.user_id)
