"""Synthetic native Doctor adapter, no live health data, programs or database."""
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from test_widget_context import fixture


def setup_care(tmp_path):
    daemon, task, skill, spec = fixture(tmp_path)
    spec["widgets"] = [{"key": "care", "title": "Treatments & care", "binding": {"type": "care"}}]
    (Path(skill.directory) / "widgets.json").write_text(json.dumps(spec))
    now = datetime.now(timezone.utc)
    def payload(member, title):
        return {"member": member, "source": "Synthetic protected records", "recordedAt": now.isoformat(), "expiresAt": (now + timedelta(minutes=5)).isoformat(), "items": [{"id": "routine-1", "definition": title, "dueAt": now.isoformat(), "adherence": "unknown", "recordedAt": None, "writeState": "none"}]}
    class Adapter:
        # Models a protected data source: always query by authenticated actor,
        # never fetch the household and filter in the client or language model.
        def __init__(self):
            self.records = {task.user: payload(task.user, "Synthetic routine A"), "other": payload("other", "Synthetic routine B")}
            self.revoked = set()
            self.calls = []
        def read_for_member(self, *, actor_user_id, skill_id):
            self.calls.append((actor_user_id, skill_id))
            if actor_user_id in self.revoked:
                raise PermissionError("revoked")
            return copy.deepcopy(self.records.get(actor_user_id))
    adapter = Adapter()
    return daemon, task, skill, adapter


def test_missing_capability_then_authorized_reads_and_revocation(tmp_path):
    daemon, task, skill, adapter = setup_care(tmp_path)
    def read(actor=task.user):
        return daemon.desktop_widget_context(actor, ["care"])["widgets"][0]
    assert read()["status"] == "unavailable"
    daemon.runtime.care_records = adapter
    first = read()
    assert first["status"] == "ready" and first["care"]["items"][0]["adherence"] == "unknown"
    assert read("other")["care"]["items"][0]["definition"] == "Synthetic routine B"
    assert first["id"] != read("other")["id"]
    # A failed workflow does not change adherence or produce optimistic success.
    item = adapter.records[task.user]["items"][0]
    item["writeState"] = "failed"
    assert read()["care"]["items"][0]["adherence"] == "unknown"
    assert read()["care"]["items"][0]["writeState"] == "failed"
    item.update(writeState="verified", adherence="recorded", recordedAt=adapter.records[task.user]["recordedAt"])
    assert read()["care"]["items"][0]["writeState"] == "verified"
    adapter.revoked.add(task.user)
    assert read()["status"] == "unavailable" and "care" not in read()
    assert read("other")["status"] == "ready"
    assert all(actor in {task.user, "other"} and sid == skill.skill_id for actor, sid in adapter.calls)
    daemon.runtime.skill_store.uninstall(skill.skill_id)
    assert daemon.desktop_widget_context(task.user, ["care"])["widgets"] == []


@pytest.mark.parametrize("change", ["cross-member", "expired", "future", "naive", "duplicate", "missing-date", "unknown-field", "oversize", "malformed", "unconfigured"])
def test_invalid_payloads_never_leak_or_invent_missed_adherence(tmp_path, change):
    daemon, task, skill, adapter = setup_care(tmp_path)
    daemon.runtime.care_records = adapter
    data = adapter.records[task.user]
    if change == "cross-member": data["member"] = "other"
    if change == "expired": data["expiresAt"] = "2000-01-01T00:00:00+00:00"
    if change == "future": data["recordedAt"] = "2100-01-01T00:00:00+00:00"
    if change == "naive": data["items"][0]["dueAt"] = "2026-01-01T00:00:00"
    if change == "duplicate": data["items"] *= 2
    if change == "missing-date": data["items"][0]["adherence"] = "missed"
    if change == "unknown-field": data["items"][0]["script"] = "unsafe"
    if change == "oversize": data["items"][0]["definition"] = "x" * 501
    if change == "malformed": data["items"] = "bad"
    if change == "unconfigured": adapter.records.clear()
    row = daemon.desktop_widget_context(task.user)["widgets"][0]
    assert row["status"] == "unavailable" and "care" not in row and not row["text"]


def test_no_program_paths_or_member_override_in_skill_binding(tmp_path):
    from alphonse.agent_v2.widget_context import definitions
    daemon, task, skill, adapter = setup_care(tmp_path)
    for extra in ({"member": "other"}, {"path": "private.sqlite"}, {"script": "doctor.py"}, {"artifact_id": "artifact.health"}):
        (Path(skill.directory) / "widgets.json").write_text(json.dumps({"version": 1, "widgets": [{"key": "care", "title": "Care", "binding": {"type": "care", **extra}}]}))
        assert definitions(skill) == []
