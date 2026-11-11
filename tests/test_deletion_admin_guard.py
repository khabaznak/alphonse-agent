"""A rejected admin deletion must not delete the admin's associated data."""

from pathlib import Path

import pytest

from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def test_rejected_admin_deletion_preserves_projects_and_files(tmp_path: Path) -> None:
    users = V2UserStore(tmp_path / "users.sqlite3")
    admin = users.onboard(display_name="Alex", users_root=tmp_path / "profiles")
    projects = ProjectStore(tmp_path / "projects.sqlite3")
    root = users.managed_project_root(admin.user_id) / "important"
    project = projects.create_project(name="Important", root_path=str(root), owner_user_id=admin.user_id)
    data = root / "synthetic.txt"
    data.write_text("keep this", encoding="utf-8")
    daemon = V2Daemon(build_runtime_host(user=admin.user_id, user_store=users, project_store=projects))

    with pytest.raises(ValueError, match="admin_user_cannot_be_deleted"):
        daemon.delete_user(admin.user_id, confirmation=admin.user_id)

    assert users.get_user(admin.user_id) is not None
    assert projects.get_project(project.project_id) is not None
    assert data.read_text(encoding="utf-8") == "keep this"
