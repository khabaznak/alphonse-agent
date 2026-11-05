"""Disposable data projections, never household programs or provider credentials."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from test_widget_context import fixture


def setup_snapshot(tmp_path, family="shopping"):
    daemon, task, skill, spec = fixture(tmp_path)
    root = Path(skill.directory)
    (root / "fixture.sh").write_text("#!/bin/sh\nexit 0\n")
    runtime = daemon.runtime
    runtime.artifact_store.register(artifact_id="artifact.fixture-list", name="Fixture list", description="Disposable", project_id="", owner_user_id=task.user, skill_id=skill.skill_id, storage_kind="skill", entrypoint_path="fixture.sh", argument_schema={"type": "object"})
    now = datetime.now(timezone.utc)
    data = {"version": 1, "updated_at": now.isoformat(), "expires_at": (now + timedelta(hours=1)).isoformat(), "items": [{"id": "milk", "title": "Milk", "status": "pending"}]}
    path = root / "list.json"
    path.write_text(json.dumps(data))
    binding = {"type": "coordination", "family": family, "artifact_id": "artifact.fixture-list", "path": "list.json"}
    spec["widgets"] = [{"key": "list", "title": "Fixture list", "binding": binding}]
    (root / "widgets.json").write_text(json.dumps(spec))
    return daemon, task, skill, spec, path, data


@pytest.mark.parametrize("family", ["todos", "shopping", "reminders", "appointments"])
def test_read_update_repeated_views_and_revocation(tmp_path, family):
    daemon, task, skill, spec, path, data = setup_snapshot(tmp_path, family)
    def read():
        return daemon.poll_desktop(client_id="fixture", user=task.user)["widget_context"]["widgets"][0]
    first = read()
    assert first["status"] == "ready" and first["coordination"]["family"] == family
    assert first["coordination"]["items"][0]["status"] == "pending"
    assert read() == first
    assert json.loads(path.read_text()) == data  # Opening never writes or runs the fixture.
    # Simulate an owning skill's confirmed effect published atomically.
    data["items"][0]["status"] = "completed"
    replacement = path.with_suffix(".new")
    replacement.write_text(json.dumps(data)); replacement.replace(path)
    second = read()
    assert second["id"] == first["id"]
    assert second["coordination"]["items"][0]["status"] == "completed"
    assert daemon.desktop_widget_context("other", [first["id"]])["widgets"] == []
    runtime = daemon.runtime
    runtime.artifact_store.set_enabled("artifact.fixture-list", False)
    denied = read()
    assert denied["status"] == "unavailable" and "coordination" not in denied
    runtime.artifact_store.set_enabled("artifact.fixture-list", True)
    path.unlink()
    assert read()["status"] == "unavailable"


@pytest.mark.parametrize("change", ["expired", "future", "duplicate", "unsafe", "naive", "oversize", "owner-spoof", "wrong-skill", "escape", "symlink"])
def test_unavailable_instead_of_untrusted_or_stale_data(tmp_path, change):
    daemon, task, skill, spec, path, data = setup_snapshot(tmp_path)
    binding = spec["widgets"][0]["binding"]
    if change == "expired": data["expires_at"] = "2000-01-01T00:00:00+00:00"
    if change == "future": data["updated_at"] = "2100-01-01T00:00:00+00:00"
    if change == "duplicate": data["items"] *= 2
    if change == "unsafe": data["items"][0]["action"] = "shell"
    if change == "naive": data["updated_at"] = "2026-01-01T00:00:00"
    if change == "oversize": data["items"] *= 51
    if change == "owner-spoof": data["owner"] = "other"
    if change == "wrong-skill": binding["artifact_id"] = "artifact.missing"
    if change == "escape": binding["path"] = "../outside.json"
    path.write_text(json.dumps(data))
    if change == "symlink":
        outside = tmp_path / "outside.json"; outside.write_text(json.dumps(data))
        path.unlink(); path.symlink_to(outside)
    (Path(skill.directory) / "widgets.json").write_text(json.dumps(spec))
    row = daemon.desktop_widget_context(task.user)["widgets"][0]
    assert row["status"] == "unavailable" and "coordination" not in row


def test_catalog_sharing_is_permission_not_relevance(tmp_path):
    from dataclasses import replace
    from alphonse.agent_v2.coordination_widgets import snapshot
    daemon, task, skill, spec, path, data = setup_snapshot(tmp_path)
    runtime = daemon.runtime
    member = runtime.user_store.create_user(display_name="Other fixture")
    binding = spec["widgets"][0]["binding"]
    assert snapshot(runtime, member.user_id, skill.skill_id, binding) is None
    record = runtime.artifact_store.get("artifact.fixture-list")
    runtime.artifact_store.update_location(replace(record, shared_with_user_ids=(member.user_id,)))
    assert snapshot(runtime, member.user_id, skill.skill_id, binding)["owner"] == task.user
    runtime.artifact_store.update_location(record)
    assert snapshot(runtime, member.user_id, skill.skill_id, binding) is None
    # A declaration in another skill cannot read the first skill's artifact.
    assert snapshot(runtime, task.user, "skill:other", binding) is None
