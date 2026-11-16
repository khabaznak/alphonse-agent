import sqlite3

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore


MESSAGE_ID = "a" * 32


def _fixture(tmp_path, *, source_prefix="inbound:", origin_owner="owner", queue_owner=None,
             checkpoint_owner=None, include_queue=True, include_checkpoint=True):
    path = tmp_path / "household.sqlite"
    store = SQLiteConversationStore(path)
    event = store.record(owner_user_id="owner", role="user" if source_prefix == "inbound:" else "assistant",
                         content="synthetic conversation", source="desktop",
                         source_message_id=source_prefix + MESSAGE_ID)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE v2_trusted_owner_ingress (message_id TEXT PRIMARY KEY, owner_id TEXT)")
        if origin_owner is not None:
            conn.execute("INSERT INTO v2_trusted_owner_ingress VALUES (?,?)", (MESSAGE_ID, origin_owner))
        if include_queue:
            conn.execute("CREATE TABLE v2_inbound_messages (message_id TEXT PRIMARY KEY, user_id TEXT, status TEXT, prompt TEXT)")
            if queue_owner is not None:
                conn.execute("INSERT INTO v2_inbound_messages VALUES (?,?,?,?)",
                             (MESSAGE_ID, queue_owner, "processing", "synthetic prompt"))
        if include_checkpoint:
            conn.execute("CREATE TABLE v2_task_checkpoints (task_id TEXT PRIMARY KEY, owner_id TEXT, status TEXT, task_state_json TEXT)")
            if checkpoint_owner is not None:
                conn.execute("INSERT INTO v2_task_checkpoints VALUES (?,?,?,?)",
                             (MESSAGE_ID, checkpoint_owner, "done", '{"synthetic":"retained"}'))
    return store, event, path


@pytest.mark.parametrize("source_prefix", ["inbound:", "outbound:owner:"])
@pytest.mark.parametrize("copy", ["queue", "checkpoint", "both"])
def test_owner_deletion_refuses_retained_request_or_checkpoint_copies(tmp_path, source_prefix, copy):
    store, event, path = _fixture(tmp_path, source_prefix=source_prefix,
                                  queue_owner="owner" if copy in {"queue", "both"} else None,
                                  checkpoint_owner="owner" if copy in {"checkpoint", "both"} else None)
    with pytest.raises(ValueError, match="owner_conversation_deletion_requires_reconciliation"):
        store.delete(event_id=event.event_id, owner_user_id="owner")
    assert store.read(event_id=event.event_id, requester_user_id="owner") is not None
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM v2_conversation_tombstones").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM v2_inbound_messages").fetchone()[0] == (copy in {"queue", "both"})
        assert conn.execute("SELECT COUNT(*) FROM v2_task_checkpoints").fetchone()[0] == (copy in {"checkpoint", "both"})


@pytest.mark.parametrize("missing", ["queue", "checkpoint"])
def test_matching_origin_with_incomplete_schema_requires_reconciliation(tmp_path, missing):
    store, event, _ = _fixture(tmp_path, include_queue=missing != "queue", include_checkpoint=missing != "checkpoint")
    with pytest.raises(ValueError, match="owner_conversation_deletion_requires_reconciliation"):
        store.delete(event_id=event.event_id, owner_user_id="owner")
    assert store.read(event_id=event.event_id, requester_user_id="owner") is not None


def test_foreign_owner_origin_cannot_block_unrelated_event(tmp_path):
    store, event, _ = _fixture(tmp_path, origin_owner="other", queue_owner="other", checkpoint_owner="other")
    assert not store.delete(event_id=event.event_id, owner_user_id="other")
    assert store.delete(event_id=event.event_id, owner_user_id="owner")


def test_origin_without_remaining_sensitive_copies_allows_logical_delete(tmp_path):
    store, event, _ = _fixture(tmp_path)
    assert store.delete(event_id=event.event_id, owner_user_id="owner")
    assert store.read(event_id=event.event_id, requester_user_id="owner") is None


@pytest.mark.parametrize("copy", ["queue", "checkpoint"])
def test_missing_origin_does_not_allow_same_owner_copy_to_be_ignored(tmp_path, copy):
    store, event, _ = _fixture(tmp_path, origin_owner=None,
                              queue_owner="owner" if copy == "queue" else None,
                              checkpoint_owner="owner" if copy == "checkpoint" else None)
    with pytest.raises(ValueError, match="owner_conversation_deletion_requires_reconciliation"):
        store.delete(event_id=event.event_id, owner_user_id="owner")
    assert store.read(event_id=event.event_id, requester_user_id="owner") is not None
