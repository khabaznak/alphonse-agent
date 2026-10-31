"""Synthetic managed services only; no credentials, provider reads or writes."""
from dataclasses import replace

import pytest

from alphonse.agent_v2.automations import EventAutomationStore, EventEnvelope
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host


class Authority:
    """Disposable stand-in for the later grant/policy/run services."""

    def __init__(self):
        self.runs = {}
        self.active = {key: True for key in ("actor", "owner", "account", "grant", "project", "policy", "run")}
        self.phases = []

    def resolve(self, worker_id, check_run_id):
        return self.runs[check_run_id]

    def authorize(self, envelope, *, actor_user_id, project_id, phase):
        self.phases.append(phase)
        return (all(self.active.values()) and self.runs.get(envelope.check_run_id) == envelope
                and actor_user_id == envelope.actor_user_id and project_id == envelope.project_id)


def envelope(**overrides):
    fields = dict(publication_id="pub1", worker_id="managed", actor_user_id="alice",
                  owner_user_id="alice", project_id="school", policy_id="policy1",
                  policy_revision=1, check_run_id="run1", resource_id="inbox",
                  connection_id="account1", grant_id="grant1", grant_revision=1,
                  event_type="mail.candidates", event_version="1",
                  occurred_at="2026-10-31T10:00:00Z", observed_at="2026-10-31T10:01:00Z")
    return EventEnvelope(**(fields | overrides))


def setup(db_path=":memory:"):
    store = EventAutomationStore(db_path)
    store.register_event_type(event_type="mail.candidates", version="1", schema={"type": "object"})
    store.register_worker(worker_id="managed", display_name="Managed mail", allowed_event_types=["mail.candidates"])
    authority = Authority()
    authority.runs["run1"] = envelope()
    store.bind_managed_adapter("managed", authority)
    return store, authority


def subscribe(store, scope=None, **overrides):
    return store.create_event_automation(**(dict(owner_user_id="alice", project_id="school", name="Check",
        prompt="Review sources", event_type="mail.candidates", event_version="1", source_scope=scope or envelope()) | overrides))


def publish(store, **overrides):
    return store.publish_managed(**(dict(worker_id="managed", check_run_id="run1", event_id="e1",
        event_type="mail.candidates", event_version="1", occurred_at="forged time",
        payload={"subject": "synthetic private metadata", "owner_user_id": "bob", "project_id": "other"}) | overrides))


def test_server_scope_precedes_payload_matching_and_prevents_account_project_fanout():
    store, authority = setup()
    good = subscribe(store)
    # All payload filters match, but neither another owner/project nor account may consume.
    for changes in (dict(actor_user_id="bob", owner_user_id="bob"), dict(project_id="other"),
                    dict(connection_id="account2", grant_id="grant2")):
        scope = envelope(**changes, check_run_id=str(len(authority.runs)), publication_id=str(len(authority.runs)))
        authority.runs[scope.check_run_id] = scope
        subscribe(store, scope, owner_user_id=scope.actor_user_id, project_id=scope.project_id)
    result = publish(store)
    assert result["matches"] == [good.automation_id]
    event = store.list_events(actor_user_id="alice")[0]
    assert event["envelope"] == envelope().__dict__
    assert event["occurred_at"] == envelope().occurred_at
    assert store.list_events(actor_user_id="bob") == []
    assert store.list_events() == []
    assert store.read_event(result["event_key"], actor_user_id="bob") is None
    assert store.expand_event(result["event_key"], actor_user_id="alice", project_id="other") is None


def test_private_publication_cannot_use_worker_id_or_payload_scope_as_credentials():
    store, _ = setup()
    values = dict(worker_id="managed", event_id="forged", event_type="mail.candidates", event_version="1",
                  occurred_at="", payload={"envelope": envelope().__dict__})
    assert store.publish(**values)["accepted"] is False
    store.register_worker(worker_id="unscoped", display_name="Forged", allowed_event_types=["mail.candidates"])
    assert store.publish(**(values | {"worker_id": "unscoped"}))["accepted"] is False
    assert publish(store, check_run_id="unknown")["accepted"] is False
    assert publish(store, event_version="2")["accepted"] is False
    with pytest.raises(PermissionError):
        store.create_event_automation(owner_user_id="alice", project_id="school", name="Unscoped",
            prompt="Review", event_type="mail.candidates", event_version="1")
    assert store.list_events(actor_user_id="alice") == []


@pytest.mark.parametrize("lifecycle", ["actor", "owner", "account", "grant", "project", "policy", "run"])
def test_lifecycle_rechecked_at_publication_dispatch_history_and_expansion(lifecycle):
    store, authority = setup()
    subscribe(store)
    authority.active[lifecycle] = False
    assert publish(store)["accepted"] is False
    authority.active[lifecycle] = True
    result = publish(store)
    authority.active[lifecycle] = False
    assert store.claim_event_executions() == []
    assert store.list_events(actor_user_id="alice") == []
    assert store.read_event(result["event_key"], actor_user_id="alice") is None
    assert "dispatch" in authority.phases
    assert "expansion" in authority.phases


@pytest.mark.parametrize("change", ["worker", "subscription", "revision", "account"])
def test_pauses_and_scope_changes_fence_pending_dispatch(change):
    store, authority = setup()
    automation = subscribe(store)
    publish(store)
    if change == "worker":
        store.register_worker(worker_id="managed", display_name="Paused", allowed_event_types=["mail.candidates"], enabled=False)
    elif change == "subscription":
        store.set_automation_enabled(automation.automation_id, False)
    elif change == "revision":
        authority.runs["run1"] = replace(envelope(), policy_revision=2)
    else:
        authority.runs["run1"] = replace(envelope(), connection_id="account2")
    assert store.claim_event_executions() == []


def test_restart_preserves_private_fence_and_fails_closed_without_services(tmp_path):
    path = tmp_path / "events.db"
    store, _ = setup(path)
    subscribe(store)
    publish(store)
    restarted = EventAutomationStore(path)
    assert restarted.is_private_type("mail.candidates")
    assert restarted.is_managed_worker("managed")
    assert publish(restarted)["accepted"] is False
    assert restarted.list_events(actor_user_id="alice") == []
    assert restarted.claim_event_executions() == []


def test_duplicate_id_cannot_disclose_another_scope_event_key():
    store, authority = setup()
    publish(store)
    authority.runs["run2"] = envelope(actor_user_id="bob", owner_user_id="bob", check_run_id="run2", publication_id="pub2")
    assert publish(store, check_run_id="run2") == {"accepted": False, "reason": "event_scope_denied"}


def test_unknown_envelope_version_is_denied_and_household_resources_are_supported():
    store, authority = setup()
    authority.runs["run1"] = replace(envelope(), schema_version=2)
    assert publish(store)["accepted"] is False
    authority.runs["run1"] = envelope(connection_id="", grant_id="", grant_revision=0, resource_id="household.sensor")
    assert publish(store)["accepted"] is True


def test_ipc_uses_local_session_identity_and_private_queue_contains_only_reference():
    store, authority = setup()
    scope = envelope(actor_user_id="local", owner_user_id="local")
    authority.runs["run1"] = scope
    subscribe(store, scope, owner_user_id="local")
    subscribe(store, scope, owner_user_id="local", name="Local private check")
    # Another owner's automation and schedule must never appear in the local catalog.
    other = envelope(actor_user_id="bob", owner_user_id="bob", check_run_id="run2", publication_id="pub2")
    authority.runs["run2"] = other
    subscribe(store, other, owner_user_id="bob", name="Bob private check")
    daemon = V2Daemon(build_runtime_host(user="local"), event_store=store)
    daemon.runtime.schedule_store.create_task(owner_user_id="bob", project_id="school", name="Bob schedule",
        prompt="Private", schedule_kind="once", run_at="2030-01-01T00:00:00+00:00")
    def ipc(method, **params):
        return daemon.ipc._dispatch({"version": 1, "method": method, "params": params})
    assert ipc("publish_event", worker_id="managed", payload={"envelope": scope.__dict__})["accepted"] is False
    result = daemon.publish_managed_event(worker_id="managed", check_run_id="run1", event_id="e1",
        event_type="mail.candidates", event_version="1", occurred_at="", payload={"subject": "private"})
    queued = daemon.runtime.queue.dequeue()
    assert queued.message.metadata["event"]["payload"] == {}
    assert queued.message.metadata["event"]["requires_expansion"] is True
    catalog = ipc("automation_catalog", actor_user_id="bob", user="bob")
    assert all(item["name"] not in ("Bob private check", "Bob schedule") for item in catalog["automations"])
    assert catalog["workers"] == [] and catalog["event_types"] == []
    assert ipc("expand_automation_event", event_key=result["event_key"], user="bob")["event"]["payload"] == {"subject": "private"}
    authority.active["grant"] = False
    assert ipc("expand_automation_event", event_key=result["event_key"])["event"] is None
    assert ipc("automation_catalog")["events"] == []
    assert ipc("automation_catalog")["automations"] == []
    with pytest.raises(PermissionError):
        ipc("register_event_worker", worker_id="managed")
    with pytest.raises(PermissionError):
        ipc("register_event_type", event_type="mail.candidates")


def test_additive_migration_preserves_legacy_history_ids_and_subscriptions(tmp_path):
    import sqlite3

    path = tmp_path / "legacy.db"
    # Actual pre-envelope table, not a fixture created by the new schema.
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE v2_event_history(event_key TEXT PRIMARY KEY, worker_id TEXT NOT NULL, source_event_id TEXT NOT NULL, event_type TEXT NOT NULL, event_version TEXT NOT NULL, occurred_at TEXT NOT NULL, payload_json TEXT NOT NULL, dispatch_count INTEGER NOT NULL, created_at TEXT NOT NULL, UNIQUE(worker_id,source_event_id)) STRICT")
        conn.execute("INSERT INTO v2_event_history VALUES('old-event','sensor','old-source','sensor.reading','1','2026-10-31T00:00:00Z','{}',0,'2026-10-31T00:00:00Z')")
    store = EventAutomationStore(path)
    store.register_event_type(event_type="sensor.reading", version="1", schema={})
    store.register_worker(worker_id="sensor", display_name="Sensor", allowed_event_types=["sensor.reading"])
    subscription = store.create_event_automation(owner_user_id="local", project_id="home", name="Sensor",
        prompt="Check", event_type="sensor.reading", event_version="1")
    restarted = EventAutomationStore(path)
    assert restarted.list_events()[0]["event_key"] == "old-event"
    assert restarted.get_automation(subscription.automation_id) == subscription
    # Reclassifying an existing type never guesses a scope for its old rows.
    restarted.bind_managed_adapter("sensor", Authority())
    assert restarted.list_events(actor_user_id="local") == []
    assert restarted.visible_automations(actor_user_id="local") == []


def test_authority_exceptions_fail_closed_and_type_pause_blocks_dispatch():
    store, authority = setup()
    subscribe(store)
    result = publish(store)
    store.register_event_type(event_type="mail.candidates", version="1", schema={}, enabled=False)
    assert store.claim_event_executions() == []
    assert store.read_event(result["event_key"], actor_user_id="alice") is None
    store.register_event_type(event_type="mail.candidates", version="1", schema={}, enabled=True)
    def unavailable(*args, **kwargs):
        raise RuntimeError("service unavailable with sensitive detail")
    authority.authorize = unavailable
    assert publish(store, event_id="new") == {"accepted": False, "reason": "event_scope_denied"}
    assert store.list_events(actor_user_id="alice") == []
