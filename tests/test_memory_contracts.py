"""Synthetic boundary tests for canonical sources and disposable projections."""
from dataclasses import replace
import json
import sqlite3
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore, legacy_ledger_events
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.memory.contracts import (
    AccessScope, ContextItem, DurableMemoryRecord, RecallResult, SourceReference,
    conversation_item, visible_unique,
)
from alphonse.agent_v2.core.memory.daily_ledger import DailyLedgerProjector
from alphonse.agent_v2.core.memory.transcript import conversation_transcript
from alphonse.agent_v2.core.tools.registry.native.memory import execute_search_memory


def record(store, **kwargs):
    values = dict(owner_user_id="alex", project_id="", role="user", content="Synthetic receipt ABC-12",
                  source="desktop", source_message_id="receipt", created_at="2026-10-16T23:59:00+00:00")
    values.update(kwargs)
    return store.record(**values)


def test_contract_roundtrip_versions_and_derived_audience():
    private = SourceReference("conversation:one", "conversation", AccessScope("alex"))
    shared = AccessScope("alex", "shared", ("sam",))
    memory = DurableMemoryRecord("memory:one", "Receipt fact", shared, (private,), "2026-10-17T00:00:00+00:00")
    assert memory.visible_to("alex") and not memory.visible_to("sam")
    assert DurableMemoryRecord.from_dict(json.loads(json.dumps(memory.to_dict()))) == memory
    result = RecallResult("receipt", "found", (ContextItem(private, "ABC-12"),))
    assert RecallResult.from_dict(json.loads(json.dumps(result.to_dict()))) == result
    with pytest.raises(ValueError):
        SourceReference.from_dict({**private.to_dict(), "schema_version": 99})
    with pytest.raises(ValueError):
        RecallResult("receipt", "empty", result.items)
    with pytest.raises(ValueError):
        AccessScope("alex", "private", ("sam",))


def test_read_recall_and_assembly_share_identity_across_midnight_and_projects():
    store = SQLiteConversationStore(":memory:")
    first = record(store)
    record(store, project_id="garden", source_message_id="garden", content="Synthetic garden note",
           created_at="2026-10-17T00:01:00+00:00")
    task = TaskState(user="alex")
    text = conversation_transcript(task, store)
    result = execute_search_memory({"query": "ABC-12"}, context=SimpleNamespace(
        task=task, conversation_store=store, memory=SimpleNamespace(search=lambda **kw: "")))
    recalled = RecallResult.from_dict(result["recall"])
    direct = conversation_item(store.read(event_id=first.event_id, requester_user_id="alex"), "alex")
    assert recalled.items[0].source.source_id == direct.source.source_id
    assert direct.source.source_id in {s["source_id"] for s in task.metadata["conversation_sources"]}
    assert len(visible_unique([direct, recalled.items[0]], "alex")) == 1
    assert "ABC-12" in text and "garden note" in text
    assert first.recorded_at and first.observed_at == first.created_at


def test_visibility_is_filtered_before_any_backend_content_is_rendered():
    store = SQLiteConversationStore(":memory:")
    private = record(store, owner_user_id="sam")
    shared = record(store, owner_user_id="sam", source_message_id="shared", content="Visible shared receipt",
                    visibility="shared", shared_with=("alex",))
    # Adversarial adapter returns another owner's records and duplicate hits.
    backend = SimpleNamespace(list_recent=lambda **kw: [private, shared, shared], search=lambda **kw: [private, shared, shared])
    task = TaskState(user="alex")
    rendered = conversation_transcript(task, backend)
    assert private.content not in rendered
    assert rendered.count(shared.content) == 1
    recall = execute_search_memory({"query": "receipt"}, context=SimpleNamespace(
        task=task, conversation_store=backend, memory=SimpleNamespace(search=lambda **kw: "")))
    assert private.content not in json.dumps(recall)
    assert len(recall["recall"]["items"]) == 1
    assert store.read(event_id=private.event_id, requester_user_id="alex") is None
    assert store.read(event_id=shared.event_id, requester_user_id="alex") == shared
    assert conversation_item(SimpleNamespace(content="Unattributed secret"), "alex") is None
    assert conversation_item(replace(shared, schema_version=99), "alex") is None


def test_legacy_import_never_invents_date_speaker_or_channel():
    store = SQLiteConversationStore(":memory:")
    events = legacy_ledger_events("### Task old\n#### Conversation\n- User: Old question\n#### Conversation\n- Alphonse: Old answer", owner_user_id="alex", project_id="")
    assert store.import_legacy_events(owner_user_id="alex", project_id="", events=events) == 2
    assert store.import_legacy_events(owner_user_id="alex", project_id="", events=events) == 0
    for event in store.list_all(owner_user_id="alex"):
        item = conversation_item(event, "alex")
        assert event.created_at == event.observed_at == event.speaker_id == ""
        assert item.source.channel is None and item.source.speaker_id is None
        assert event.recorded_at and item.source.provenance == "legacy_import"
        assert store.read(event_id=event.event_id, requester_user_id="alex") == event


def test_additive_migration_preserves_original_events_and_ambiguous_imports(tmp_path):
    db = tmp_path / "old.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE v2_conversation_events (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
            owner_user_id TEXT NOT NULL, project_id TEXT NOT NULL DEFAULT '',
            memory_session_id TEXT NOT NULL DEFAULT '', role TEXT NOT NULL, content TEXT NOT NULL,
            source TEXT NOT NULL, source_message_id TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL) STRICT""")
        conn.execute("INSERT INTO v2_conversation_events VALUES (7,'old','alex','','','assistant','Original answer','telegram','legacy:alex::1','2026-10-16T12:00:00+00:00')")
    first = SQLiteConversationStore(db).list_all(owner_user_id="alex")[0]
    assert (first.sequence, first.event_id, first.content, first.source, first.created_at) == (
        7, "old", "Original answer", "telegram", "2026-10-16T12:00:00+00:00")
    assert first.observed_at == first.recorded_at == first.speaker_id == ""
    assert first.provenance == "legacy_import"
    restarted = SQLiteConversationStore(db)
    assert restarted.list_all(owner_user_id="alex")[0] == first
    interval = dict(owner_user_id="alex", local_date="2026-10-16",
                    start_utc="2026-10-16T00:00:00+00:00", end_utc="2026-10-17T00:00:00+00:00")
    assert not restarted.user_has_unseen_messages(**interval)
    assert restarted.mark_user_seen(**interval) == 0


def test_rebuild_and_erasure_preserve_ids_remove_all_generated_copies(tmp_path):
    store = SQLiteConversationStore(tmp_path / "history.sqlite")
    projector = DailyLedgerProjector(conversation_store=store, users_root=lambda: tmp_path,
                                     timezone_provider=lambda: "UTC")
    store.set_daily_ledger_projector(projector)
    erased = record(store)
    retained = record(store, source_message_id="today", content="Retained note", created_at="2026-10-17T00:01:00+00:00")
    before = store.list_all(owner_user_id="alex")
    projector.rebuild_owner("alex")
    projector.rebuild_owner("alex")
    assert store.list_all(owner_user_id="alex") == before
    assert not store.delete(event_id=erased.event_id, owner_user_id="sam")
    assert store.delete(event_id=erased.event_id, owner_user_id="alex")
    assert store.read(event_id=erased.event_id, requester_user_id="alex") is None
    assert store.search(owner_user_id="alex", query="ABC-12") == []
    assert record(store) is None  # Source redelivery cannot resurrect erased history.
    root = tmp_path / "alex/memory/daily"
    assert all(erased.content not in path.read_text() for path in root.glob("*.md"))
    assert any(retained.content in path.read_text() for path in root.glob("*.md"))
    restarted = SQLiteConversationStore(tmp_path / "history.sqlite")
    assert restarted.list_all(owner_user_id="alex") == [retained]
    assert record(restarted) is None


def test_erasure_projection_failure_leaves_no_stale_generated_text(tmp_path, monkeypatch):
    store = SQLiteConversationStore(":memory:")
    projector = DailyLedgerProjector(conversation_store=store, users_root=lambda: tmp_path)
    store.set_daily_ledger_projector(projector)
    event = record(store)
    def broken(*args):
        raise OSError("synthetic projection failure")
    monkeypatch.setattr(projector, "rebuild_owner", broken)
    with pytest.raises(OSError, match="synthetic"):
        store.delete(event_id=event.event_id, owner_user_id="alex")
    assert store.read(event_id=event.event_id, requester_user_id="alex") is None
    assert not list((tmp_path / "alex/memory/daily").glob("*.md"))


def test_recall_envelope_distinguishes_empty_and_unavailable_backends():
    def unavailable(**kwargs):
        raise OSError("offline")
    def run(search):
        return execute_search_memory({"query": "receipt"}, context=SimpleNamespace(
            task=TaskState(user="alex"), conversation_store=SimpleNamespace(search=search),
            memory=SimpleNamespace(search=lambda **kw: "")))["recall"]["status"]
    assert run(lambda **kw: []) == "empty"
    assert run(unavailable) == "unavailable"


def test_derived_memory_rechecks_current_sources_after_erasure():
    from alphonse.agent_v2.core.memory.contracts import resolve_durable_memory
    store = SQLiteConversationStore(":memory:")
    event = record(store)
    source = conversation_item(event, "alex").source
    memory = DurableMemoryRecord("memory:receipt", "Derived ABC-12", AccessScope("alex"), (source,), event.recorded_at)
    def resolve(ref, requester):
        current = store.read(event_id=ref.source_id.removeprefix("conversation:"), requester_user_id=requester)
        return conversation_item(current, requester)
    assert resolve_durable_memory(memory, "alex", resolve) == memory
    store.delete(event_id=event.event_id, owner_user_id="alex")
    assert resolve_durable_memory(memory, "alex", resolve) is None
    with pytest.raises(ValueError, match="schema_version"):
        SourceReference.from_dict({k: v for k, v in source.to_dict().items() if k != "schema_version"})


def test_task_evidence_projection_is_owner_scoped_and_preserves_attempts():
    task = TaskState(user="alex", task_id="synthetic-task", evidence_journal=[
        {"evidence_ref": "tool-call:one", "status": "failed", "result": {"error": "timeout"}},
        {"evidence_ref": "tool-call:one", "status": "success", "result": {"value": 12}},
    ])
    before = task.to_dict()
    projected = task.evidence_items("alex")
    assert [item.source.source_id for item in projected] == ["task-evidence:synthetic-task:0", "task-evidence:synthetic-task:1"]
    assert all(item.representation == "audit" for item in projected)
    assert task.evidence_items("sam") == []
    assert TaskState.from_dict(before).evidence_items("alex") == projected
    assert task.to_dict() == before


def test_historical_reference_cannot_preserve_revoked_sharing():
    from alphonse.agent_v2.core.memory.contracts import resolve_durable_memory
    store = SQLiteConversationStore(":memory:")
    event = record(store, owner_user_id="sam", visibility="shared", shared_with=("alex",))
    source = conversation_item(event, "alex").source
    memory = DurableMemoryRecord("memory:shared-receipt", "ABC-12", AccessScope("alex"), (source,), event.recorded_at)
    # Simulate the owner revoking the persisted grant; the old snapshot survives.
    with store._connect() as conn:
        conn.execute("UPDATE v2_conversation_events SET visibility='private',shared_with='[]' WHERE event_id=?", (event.event_id,))
    def resolve(ref, requester):
        return conversation_item(store.read(event_id=event.event_id, requester_user_id=requester), requester)
    assert memory.visible_to("alex")  # Cached provenance alone is insufficient.
    assert resolve_durable_memory(memory, "alex", resolve) is None


def test_recording_an_everyday_message_requires_no_project():
    store = SQLiteConversationStore(":memory:")
    event = store.record(owner_user_id="alex", role="user", content="Hello", source="desktop", source_message_id="hello")
    assert event.project_id == ""
    assert store.list_recent(owner_user_id="alex") == [event]


@pytest.mark.parametrize("recipients", ["alex", {"alex": True}, None, 42])
def test_malformed_recipients_are_never_interpreted_as_grants(recipients):
    with pytest.raises(ValueError, match="recipients"):
        AccessScope.from_dict({"owner_user_id": "sam", "visibility": "shared", "shared_with": recipients})
    store = SQLiteConversationStore(":memory:")
    event = record(store, owner_user_id="sam", visibility="shared", shared_with=("alex",))
    assert conversation_item(replace(event, shared_with=recipients), "a") is None
    with store._connect() as conn:
        conn.execute("UPDATE v2_conversation_events SET shared_with=? WHERE event_id=?", (json.dumps(recipients), event.event_id))
    with pytest.raises(ValueError, match="recipients"):
        store.read(event_id=event.event_id, requester_user_id="a")
