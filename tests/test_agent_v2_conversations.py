from __future__ import annotations

import sqlite3

from alphonse.agent_v2.conversations import SQLiteConversationStore, legacy_ledger_events
from alphonse.agent_v2.core.messages import CommunicationChannel, InMemoryMessageQueue
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host


def test_channel_records_project_user_turn_once() -> None:
    store = SQLiteConversationStore(":memory:")
    channel = CommunicationChannel(InMemoryMessageQueue(), conversation_store=store)

    queued = channel.queue_message(prompt="Hello from Telegram", user="alex", project_id="innovator", integration_id="telegram-home", provider_key="telegram", message_id="message-1")
    channel.queue_message(prompt="Hello from Telegram", user="alex", project_id="innovator", integration_id="telegram-home", provider_key="telegram", message_id=queued.message_id)

    events = store.list(owner_user_id="alex", project_id="innovator")
    assert [(event.role, event.content, event.source) for event in events] == [("user", "Hello from Telegram", "telegram-home")]


def test_desktop_history_combines_cross_channel_timeline_in_order() -> None:
    store = SQLiteConversationStore(":memory:")
    runtime = build_runtime_host(conversation_store=store)
    daemon = V2Daemon(runtime)
    store.record(owner_user_id="alex", project_id="innovator", role="user", content="What should we build?", source="telegram-home", source_message_id="telegram:1", created_at="2026-01-01T00:00:00+00:00")
    store.record(owner_user_id="alex", project_id="innovator", role="assistant", content="Start with a prototype.", source="desktop", source_message_id="outbound:1", created_at="2026-01-01T00:01:00+00:00")
    store.record(owner_user_id="alex", project_id="other", role="assistant", content="Do not show me.", source="telegram-home", source_message_id="telegram:2", created_at="2026-01-01T00:02:00+00:00")

    history = daemon.desktop_conversation_history(user="alex", project_id="innovator")

    assert [(item["role"], item["content"], item["source"]) for item in history] == [("user", "What should we build?", "telegram-home"), ("assistant", "Start with a prototype.", "desktop")]
    assert [item["id"] for item in history] == ["telegram:1", "1"]


def test_desktop_history_refresh_sees_telegram_turns_without_daemon_restart() -> None:
    store = SQLiteConversationStore(":memory:")
    runtime = build_runtime_host(conversation_store=store)
    daemon = V2Daemon(runtime)
    store.record(owner_user_id="alex", project_id="innovator", role="assistant", content="Earlier response.", source="desktop", source_message_id="desktop:1")

    first_history = daemon.desktop_conversation_history(user="alex", project_id="innovator")
    runtime.channel.queue_message(
        prompt="Telegram arrived after the first project visit.",
        user="alex",
        project_id="innovator",
        integration_id="telegram-home",
        provider_key="telegram",
        message_id="telegram:1",
    )
    refreshed_history = daemon.desktop_conversation_history(user="alex", project_id="innovator")

    assert [item["content"] for item in first_history] == ["Earlier response."]
    assert [(item["content"], item["source"]) for item in refreshed_history] == [
        ("Earlier response.", "desktop"),
        ("Telegram arrived after the first project visit.", "telegram-home"),
    ]


def test_timestamps_are_canonical_utc_and_sort_true_chronological_order() -> None:
    store = SQLiteConversationStore(":memory:")
    store.record(owner_user_id="alex", project_id="vacations", role="user", content="Pack sunglasses", source="desktop", source_message_id="user-1", created_at="2026-07-24T19:21:42-06:00")
    store.record(owner_user_id="alex", project_id="vacations", role="assistant", content="Added.", source="desktop", source_message_id="assistant-1", created_at="2026-07-25T01:22:39+00:00")

    events = store.list(owner_user_id="alex", project_id="vacations")

    assert [(event.role, event.content) for event in events] == [("user", "Pack sunglasses"), ("assistant", "Added.")]
    assert [event.created_at for event in events] == ["2026-07-25T01:21:42+00:00", "2026-07-25T01:22:39+00:00"]


def test_same_timestamp_order_is_stable() -> None:
    store = SQLiteConversationStore(":memory:")
    for source_message_id, content in (("one", "First"), ("two", "Second")):
        store.record(owner_user_id="alex", project_id="vacations", role="user", content=content, source="desktop", source_message_id=source_message_id, created_at="2026-07-24T19:21:42-06:00")

    first_read = store.list(owner_user_id="alex", project_id="vacations")
    second_read = store.list(owner_user_id="alex", project_id="vacations")

    assert [event.event_id for event in first_read] == [event.event_id for event in second_read]
    assert [event.sequence for event in first_read] == [1, 2]
    assert [event.content for event in first_read] == ["First", "Second"]
    assert [event.created_at for event in first_read] == ["2026-07-25T01:21:42+00:00"] * 2


def test_project_unread_cursor_is_authoritative_and_project_scoped() -> None:
    store = SQLiteConversationStore(":memory:")
    first = store.record(owner_user_id="alex", project_id="alpha", role="user", content="Alpha one", source="desktop", source_message_id="inbound:alpha-1")
    store.record(owner_user_id="alex", project_id="beta", role="assistant", content="Beta one", source="telegram", source_message_id="outbound:beta-1")

    assert store.project_unread_counts(owner_user_id="alex") == {"alpha": 1, "beta": 1}
    store.mark_project_seen(owner_user_id="alex", project_id="alpha", through_sequence=first.sequence)
    assert store.project_unread_counts(owner_user_id="alex") == {"beta": 1}


def test_legacy_sequence_migration_initializes_existing_history_as_seen(tmp_path) -> None:
    db_path = tmp_path / "legacy-conversations.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE v2_conversation_events (
              event_id TEXT PRIMARY KEY,
              owner_user_id TEXT NOT NULL,
              project_id TEXT NOT NULL DEFAULT '',
              role TEXT NOT NULL,
              content TEXT NOT NULL,
              source TEXT NOT NULL,
              source_message_id TEXT NOT NULL DEFAULT '' UNIQUE,
              created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO v2_conversation_events VALUES (?,?,?,?,?,?,?,?)",
            ("event-b", "alex", "alpha", "assistant", "First inserted", "desktop", "outbound:first", "2026-01-01T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO v2_conversation_events VALUES (?,?,?,?,?,?,?,?)",
            ("event-a", "alex", "alpha", "user", "Second inserted", "telegram", "inbound:second", "2026-01-01T00:00:00+00:00"),
        )

    store = SQLiteConversationStore(db_path)
    events = store.list(owner_user_id="alex", project_id="alpha")

    assert [(event.sequence, event.content) for event in events] == [(1, "First inserted"), (2, "Second inserted")]
    assert store.project_unread_counts(owner_user_id="alex") == {}
    store.record(owner_user_id="alex", project_id="alpha", role="assistant", content="New", source="desktop", source_message_id="outbound:new")
    assert store.project_unread_counts(owner_user_id="alex") == {"alpha": 1}


def test_store_startup_migrates_legacy_offset_timestamps_idempotently(tmp_path) -> None:
    db_path = tmp_path / "conversations.sqlite3"
    store = SQLiteConversationStore(db_path)
    store.record(owner_user_id="alex", project_id="vacations", role="user", content="First", source="desktop", source_message_id="first", created_at="2026-07-24T18:00:00+00:00")
    store.record(owner_user_id="alex", project_id="vacations", role="assistant", content="Second", source="desktop", source_message_id="second", created_at="2026-07-24T18:01:00+00:00")
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE v2_conversation_events SET created_at=? WHERE source_message_id='first'", ("2026-07-24T12:00:00-06:00",))
        conn.execute("UPDATE v2_conversation_events SET created_at=? WHERE source_message_id='second'", ("2026-07-24T18:01:00+00:00",))

    migrated = SQLiteConversationStore(db_path)
    first_pass = migrated.list(owner_user_id="alex", project_id="vacations")
    restarted = SQLiteConversationStore(db_path)
    second_pass = restarted.list(owner_user_id="alex", project_id="vacations")

    assert [event.created_at for event in first_pass] == ["2026-07-24T18:00:00+00:00", "2026-07-24T18:01:00+00:00"]
    assert [event.created_at for event in second_pass] == ["2026-07-24T18:00:00+00:00", "2026-07-24T18:01:00+00:00"]


def test_legacy_ledger_recovers_only_visible_turns() -> None:
    ledger = """# Memory Ledger
### Task one
#### Conversation
- User: What is this?
#### Plan
- internal: hidden
#### Conversation
- Alphonse: It is a prototype.
#### Tool Result
- hidden
"""

    events = legacy_ledger_events(ledger, owner_user_id="alex", project_id="innovator")

    assert [(event["role"], event["content"]) for event in events] == [("user", "What is this?"), ("assistant", "It is a prototype.")]


def test_desktop_history_uses_legacy_ledger_when_timeline_is_empty(monkeypatch) -> None:
    runtime = build_runtime_host(conversation_store=SQLiteConversationStore(":memory:"))
    daemon = V2Daemon(runtime)
    monkeypatch.setattr(runtime.core.memory, "latest_content", lambda **_: "### Task one\n#### Conversation\n- User: Restore me\n#### Conversation\n- Alphonse: Restored.")

    history = daemon.desktop_conversation_history(user="alex", project_id="innovator")

    assert [(item["role"], item["content"]) for item in history] == [("user", "Restore me"), ("assistant", "Restored.")]
