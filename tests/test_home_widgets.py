"""Disposable home sources exercise the real Desktop poll and catalog boundary."""
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from test_coordination_widgets import setup_snapshot
from alphonse.agent_v2.home_widgets import snapshot


def setup_home(tmp_path, family="sensors"):
    daemon, task, skill, spec, path, data = setup_snapshot(tmp_path)
    spec["widgets"][0]["binding"].update(type="home", family=family)
    now = datetime.now(timezone.utc)
    data["updated_at"] = now.isoformat()
    data["items"] = ([{"id": "fixture-temperature", "name": "Fixture room", "source": "Fixture provider", "unit": "°C", "value": 23.5, "observed_at": now.isoformat(), "stale_after_seconds": 60, "state": "connected"}] if family == "sensors" else [{"id": "fixture-contact", "name": "Fixture contact", "role": "Chosen contact", "phone": "+1 555 0100", "email": "fixture@example.invalid"}])
    path.write_text(json.dumps(data))
    (Path(skill.directory) / "widgets.json").write_text(json.dumps(spec))
    return daemon, task, skill, spec, path, data


def read(daemon, task):
    return daemon.poll_desktop(client_id="home-fixture", user=task.user)["widget_context"]["widgets"][0]


@pytest.mark.parametrize("family", ["sensors", "emergency_contacts"])
def test_poll_updates_and_revocation_never_write_or_execute(tmp_path, family):
    daemon, task, skill, spec, path, data = setup_home(tmp_path, family)
    before = {p: p.read_bytes() for p in Path(skill.directory).iterdir() if p.is_file()}
    first = read(daemon, task)
    assert first["status"] == "ready" and first["home"]["family"] == family
    assert read(daemon, task) == first
    assert before == {p: p.read_bytes() for p in before}
    data["items"][0]["name"] = "Updated fixture"
    path.write_text(json.dumps(data))
    assert read(daemon, task)["home"]["items"][0]["name"] == "Updated fixture"
    assert read(daemon, task)["id"] == first["id"]
    daemon.runtime.artifact_store.set_enabled("artifact.fixture-list", False)
    denied = read(daemon, task)
    assert denied["status"] == "unavailable" and "home" not in denied
    daemon.runtime.artifact_store.set_enabled("artifact.fixture-list", True)
    path.unlink()
    assert read(daemon, task)["status"] == "unavailable"
    daemon.runtime.skill_store.uninstall(skill.skill_id)
    assert daemon.desktop_widget_context(task.user, [first["id"]])["widgets"] == []


@pytest.mark.parametrize("case", ["old-reading", "expired-source", "stale", "disconnected", "error", "no-reading"])
def test_sensor_freshness_is_explicit(tmp_path, case):
    daemon, task, skill, spec, path, data = setup_home(tmp_path)
    sensor = data["items"][0]
    if case == "old-reading": sensor["observed_at"] = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
    if case == "expired-source":
        data["updated_at"] = sensor["observed_at"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        data["expires_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    if case in ("stale", "disconnected", "error"): sensor["state"] = case
    if case == "no-reading": sensor.update(state="disconnected", value=None, observed_at=None)
    path.write_text(json.dumps(data))
    payload = read(daemon, task)["home"]
    assert payload["items"][0]["state"] == ({"no-reading": "disconnected"}.get(case, case) if case in ("stale", "disconnected", "error", "no-reading") else "stale")


@pytest.mark.parametrize("case", ["future", "naive", "nan", "bool", "missing-unit", "duplicate", "unknown", "bad-state", "no-time", "no-value", "large", "traversal", "symlink", "wrong-skill", "long-lifetime", "bad-shape"])
def test_untrusted_sensors_are_unavailable(tmp_path, case):
    daemon, task, skill, spec, path, data = setup_home(tmp_path)
    item = data["items"][0]
    binding = spec["widgets"][0]["binding"]
    if case == "future": item["observed_at"] = "2100-01-01T00:00:00Z"
    if case == "naive": item["observed_at"] = "2026-01-01T00:00:00"
    if case == "nan": item["value"] = float("nan")
    if case == "bool": item["value"] = True
    if case == "missing-unit": item["unit"] = ""
    if case == "duplicate": data["items"] *= 2
    if case == "unknown": item["command"] = "call"
    if case == "bad-state": item["state"] = []
    if case == "no-time": item["observed_at"] = None
    if case == "no-value": item["value"] = None
    if case == "large": data["items"] *= 51
    if case == "traversal": binding["path"] = "../escape.json"
    if case == "wrong-skill": binding["artifact_id"] = "artifact.missing"
    if case == "long-lifetime": data["expires_at"] = "2100-01-01T00:00:00Z"
    if case == "bad-shape": data = []
    path.write_text(json.dumps(data))
    if case == "symlink":
        outside = tmp_path / "escape.json"; outside.write_text(json.dumps(data))
        path.unlink(); path.symlink_to(outside)
    (Path(skill.directory) / "widgets.json").write_text(json.dumps(spec))
    row = read(daemon, task)
    assert row["status"] == "unavailable" and "home" not in row


@pytest.mark.parametrize("case", ["expired", "phone-code", "email-code", "role", "duplicate", "unknown", "empty", "no-actions"])
def test_contacts_are_configured_data_only(tmp_path, case):
    daemon, task, skill, spec, path, data = setup_home(tmp_path, "emergency_contacts")
    item = data["items"][0]
    if case == "expired":
        data["updated_at"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        data["expires_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    if case == "phone-code": item["phone"] = "javascript:alert(1)"
    if case == "email-code": item["email"] = "x@example.invalid\nBcc:someone"
    if case == "role": item["role"] = ""
    if case == "duplicate": data["items"] *= 2
    if case == "unknown": item["url"] = "tel:911"
    if case == "empty": data["items"] = []
    if case == "no-actions": item.update(phone=None, email=None)
    path.write_text(json.dumps(data))
    row = read(daemon, task)
    assert row["status"] == ("ready" if case in ("empty", "no-actions") else "unavailable")
    if row["status"] == "unavailable": assert "home" not in row


@pytest.mark.parametrize("family", ["sensors", "emergency_contacts"])
def test_sharing_and_skill_ownership_are_rechecked(tmp_path, family):
    daemon, task, skill, spec, path, data = setup_home(tmp_path, family)
    runtime = daemon.runtime
    actor = runtime.user_store.create_user(display_name="Other fixture").user_id
    binding = spec["widgets"][0]["binding"]
    assert snapshot(runtime, actor, skill.skill_id, binding) is None
    record = runtime.artifact_store.get(binding["artifact_id"])
    runtime.artifact_store.update_location(replace(record, shared_with_user_ids=(actor,)))
    assert snapshot(runtime, actor, skill.skill_id, binding)["owner"] == task.user
    assert snapshot(runtime, actor, "skill:another", binding) is None
    runtime.artifact_store.update_location(record)
    assert snapshot(runtime, actor, skill.skill_id, binding) is None
    assert daemon.desktop_widget_context(actor, [read(daemon, task)["id"]])["widgets"] == []
