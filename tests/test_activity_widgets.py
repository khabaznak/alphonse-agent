"""UI integration fixtures only; trainer implementation remains issue #4."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from test_coordination_widgets import setup_snapshot


def activity_fixture(tmp_path, family="progress"):
    daemon, task, skill, spec, path, data = setup_snapshot(tmp_path)
    now = datetime.fromisoformat(data["updated_at"])
    item = {"id": "yesterday", "title": "Yesterday's exercise", "scope": "session", "recorded_at": (now - timedelta(days=1)).isoformat(), "status": "active", "value": 3, "target": 5, "unit": "sets", "display": "dots"}
    if family == "timer":
        item = {"id": "timer", "title": "Exercise interval", "scope": "session", "recorded_at": now.isoformat(), "status": "running", "elapsed_ms": 10000, "duration_ms": 60000, "running_since": now.isoformat()}
    data["items"] = [item]
    spec["widgets"][0]["binding"].update(type="activity", family=family)
    (Path(skill.directory) / "widgets.json").write_text(json.dumps(spec))
    path.write_text(json.dumps(data))
    return daemon, task, skill, spec, path, data


@pytest.mark.parametrize("family", ["progress", "tracker", "timer"])
def test_poll_source_updates_restart_retention_and_access(tmp_path, family):
    daemon, task, skill, spec, path, data = activity_fixture(tmp_path, family)
    def read():
        return daemon.poll_desktop(client_id="fixture", user=task.user)["widget_context"]["widgets"][0]
    first = read()
    assert first["status"] == "ready" and first["activity"]["items"] == data["items"]
    assert read() == first
    from alphonse.agent_v2.daemon import V2Daemon
    restarted = V2Daemon(daemon.runtime)
    assert restarted.desktop_widget_context(task.user, [first["id"]])["widgets"][0]["activity"] == first["activity"]
    assert json.loads(path.read_text()) == data
    daemon.kill_switch.clear()
    # Retained identities survive loss of active conversational context.
    retained = daemon.desktop_widget_context(task.user, [first["id"]])["widgets"][0]
    assert retained["activity"] == first["activity"]
    daemon.kill_switch.activate(message_id=task.message_id, task_id=task.task_id, user_id=task.user, project_id="", metadata={})
    data["items"][0]["status"] = "completed"
    if family == "timer":
        data["items"][0].update(running_since=None, elapsed_ms=45000)
    else:
        data["items"][0]["value"] = 4  # Completed source can differ from planned quantity.
    path.write_text(json.dumps(data))
    assert read()["activity"]["items"] == data["items"]
    assert daemon.desktop_widget_context("other", [first["id"]])["widgets"] == []
    daemon.runtime.artifact_store.set_enabled("artifact.fixture-list", False)
    denied = read()
    assert denied["status"] == "unavailable" and "activity" not in denied


@pytest.mark.parametrize("change", ["expired", "escape", "symlink", "wrong-skill", "nan", "boolean", "zero-target", "too-many-dots", "duplicate", "future", "naive", "extra", "bad-scope"])
def test_invalid_activity_fails_closed(tmp_path, change):
    daemon, task, skill, spec, path, data = activity_fixture(tmp_path)
    item = data["items"][0]
    if change == "expired": data["expires_at"] = "2000-01-01T00:00:00Z"
    if change == "escape": spec["widgets"][0]["binding"]["path"] = "../outside.json"
    if change == "wrong-skill": spec["widgets"][0]["binding"]["artifact_id"] = "artifact.missing"
    if change == "nan": item["value"] = float("nan")
    if change == "boolean": item["value"] = True
    if change == "zero-target": item["target"] = 0
    if change == "too-many-dots": item["target"] = 101
    if change == "duplicate": data["items"] *= 2
    if change == "future": item["recorded_at"] = "2100-01-01T00:00:00Z"
    if change == "naive": item["recorded_at"] = "2026-01-01T00:00:00"
    if change == "extra": item["execute"] = "shell"
    if change == "bad-scope": item["scope"] = []
    path.write_text(json.dumps(data))
    if change == "symlink":
        outside = tmp_path / "outside.json"; outside.write_text(json.dumps(data))
        path.unlink(); path.symlink_to(outside)
    (Path(skill.directory) / "widgets.json").write_text(json.dumps(spec))
    row = daemon.desktop_widget_context(task.user)["widgets"][0]
    assert row["status"] == "unavailable" and "activity" not in row


@pytest.mark.parametrize("change", ["paused", "resume", "cancelled", "bad-running", "bad-paused", "future", "negative"])
def test_persisted_timer_states(tmp_path, change):
    daemon, task, skill, spec, path, data = activity_fixture(tmp_path, "timer")
    item = data["items"][0]
    if change in {"paused", "cancelled"}: item.update(status=change, running_since=None, elapsed_ms=25000)
    if change == "resume": item.update(elapsed_ms=25000, running_since=data["updated_at"])
    if change == "bad-running": item["running_since"] = None
    if change == "bad-paused": item["status"] = "paused"
    if change == "future": item["running_since"] = "2100-01-01T00:00:00Z"
    if change == "negative": item["elapsed_ms"] = -1
    path.write_text(json.dumps(data))
    row = daemon.desktop_widget_context(task.user)["widgets"][0]
    assert (row["status"] == "ready") == (change in {"paused", "cancelled", "resume"})
