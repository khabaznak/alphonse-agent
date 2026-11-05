from __future__ import annotations

import sqlite3

from alphonse.agent_v2.conversations import SQLiteConversationStore, legacy_ledger_events
from alphonse.agent_v2.core.messages import CommunicationChannel, InMemoryMessageQueue
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.core.messages import SQLiteMessageQueue
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.users import V2UserStore


def test_channel_records_project_user_turn_once() -> None:
    store = SQLiteConversationStore(":memory:")
    channel = CommunicationChannel(InMemoryMessageQueue(), conversation_store=store)

    queued = channel.queue_message(prompt="Hello from Telegram", user="alex", project_id="innovator", integration_id="telegram-home", provider_key="telegram", message_id="message-1")
    channel.queue_message(prompt="Hello from Telegram", user="alex", project_id="innovator", integration_id="telegram-home", provider_key="telegram", message_id=queued.message_id)

    events = store.list(owner_user_id="alex", project_id="innovator")
    assert [(event.role, event.content, event.source) for event in events] == [("user", "Hello from Telegram", "telegram-home")]


def test_desktop_history_combines_all_projects_for_one_local_day_in_order() -> None:
    store = SQLiteConversationStore(":memory:")
    runtime = build_runtime_host(conversation_store=store)
    daemon = V2Daemon(runtime)
    store.record(owner_user_id="alex", project_id="innovator", role="user", content="What should we build?", source="telegram-home", source_message_id="telegram:1", created_at="2026-01-01T00:00:00+00:00")
    store.record(owner_user_id="alex", project_id="innovator", role="assistant", content="Start with a prototype.", source="desktop", source_message_id="outbound:1", created_at="2026-01-01T00:01:00+00:00")
    store.record(owner_user_id="alex", project_id="other", role="assistant", content="Another project reply.", source="telegram-home", source_message_id="telegram:2", created_at="2026-01-01T00:02:00+00:00")
    store.record(owner_user_id="alex", project_id="", role="user", content="Unassociated question.", source="desktop", source_message_id="desktop:3", created_at="2026-01-01T00:03:00+00:00")

    history = daemon.desktop_conversation_history(user="alex", project_id="innovator", local_date="2026-01-01")

    assert [(item["role"], item["content"], item["source"]) for item in history] == [("user", "What should we build?", "telegram-home"), ("assistant", "Start with a prototype.", "desktop"), ("assistant", "Another project reply.", "telegram-home"), ("user", "Unassociated question.", "desktop")]
    assert [item["id"] for item in history] == ["telegram:1", "1", "telegram:2", "desktop:3"]


def test_desktop_history_refresh_sees_telegram_turns_without_daemon_restart() -> None:
    store = SQLiteConversationStore(":memory:")
    runtime = build_runtime_host(conversation_store=store)
    daemon = V2Daemon(runtime)
    store.record(owner_user_id="alex", project_id="innovator", role="assistant", content="Earlier response.", source="desktop", source_message_id="desktop:1")

    first_history = daemon.desktop_conversation_history(user="alex")
    runtime.channel.queue_message(
        prompt="Telegram arrived after the first project visit.",
        user="alex",
        project_id="innovator",
        integration_id="telegram-home",
        provider_key="telegram",
        message_id="telegram:1",
    )
    refreshed_history = daemon.desktop_conversation_history(user="alex")

    assert [item["content"] for item in first_history] == ["Earlier response."]
    assert [(item["content"], item["source"]) for item in refreshed_history] == [
        ("Earlier response.", "desktop"),
        ("Telegram arrived after the first project visit.", "telegram-home"),
    ]


def test_desktop_history_uses_users_timezone_at_day_boundary() -> None:
    store = SQLiteConversationStore(":memory:")
    runtime = build_runtime_host(conversation_store=store)
    runtime.user_store.set_timezone("America/Mexico_City", user_id="alex")
    daemon = V2Daemon(runtime)
    store.record(owner_user_id="alex", project_id="", role="user", content="Before local midnight", source="test", source_message_id="before", created_at="2026-10-12T05:59:59+00:00")
    store.record(owner_user_id="alex", project_id="one", role="assistant", content="After local midnight", source="test", source_message_id="after", created_at="2026-10-12T06:00:00+00:00")

    assert [item["content"] for item in daemon.desktop_conversation_history(user="alex", local_date="2026-10-11")] == ["Before local midnight"]
    assert [item["content"] for item in daemon.desktop_conversation_history(user="alex", local_date="2026-10-12")] == ["After local midnight"]


def test_user_seen_cursor_and_history_survive_store_restart(tmp_path) -> None:
    path = tmp_path / "timeline.sqlite3"
    store = SQLiteConversationStore(path)
    event = store.record(owner_user_id="alex", project_id="project-a", role="assistant", content="Kept on restart", source="test", source_message_id="restart-1")
    assert store.user_has_unseen_messages(owner_user_id="alex")
    store.mark_user_seen(owner_user_id="alex", through_sequence=event.sequence)
    restarted = SQLiteConversationStore(path)

    assert not restarted.user_has_unseen_messages(owner_user_id="alex")
    assert [item.content for item in restarted.list_owner(owner_user_id="alex")] == ["Kept on restart"]


def test_seen_cursor_is_scoped_to_user_and_local_day() -> None:
    store = SQLiteConversationStore(":memory:")
    yesterday = store.record(owner_user_id="alex", project_id="", role="user", content="Yesterday", source="test", source_message_id="day-1", created_at="2026-10-11T16:00:00+00:00")
    today = store.record(owner_user_id="alex", project_id="travel", role="assistant", content="Today", source="test", source_message_id="day-2", created_at="2026-10-12T16:00:00+00:00")
    store.mark_user_seen(owner_user_id="alex", local_date="2026-10-11", through_sequence=yesterday.sequence, start_utc="2026-10-11T00:00:00+00:00", end_utc="2026-10-12T00:00:00+00:00")

    assert not store.user_has_unseen_messages(owner_user_id="alex", local_date="2026-10-11", start_utc="2026-10-11T00:00:00+00:00", end_utc="2026-10-12T00:00:00+00:00")
    assert store.user_has_unseen_messages(owner_user_id="alex", local_date="2026-10-12", start_utc="2026-10-12T00:00:00+00:00", end_utc="2026-10-13T00:00:00+00:00")
    assert today.sequence > yesterday.sequence


def test_blank_project_migration_keeps_unassociated_history_unassociated(tmp_path) -> None:
    db_path = tmp_path / "shared.sqlite3"
    users = V2UserStore(db_path)
    users.onboard(display_name="Alex", users_root=tmp_path / "profiles", user_id="alex")
    conversations = SQLiteConversationStore(db_path)
    conversations.record(owner_user_id="alex", project_id="", role="user", content="Unassociated", source="test", source_message_id="blank-project")
    conversations.record(owner_user_id="alex", project_id="garden", role="assistant", content="Garden task", source="test", source_message_id="known-project")
    runtime = build_runtime_host(user_store=users, messages=SQLiteMessageQueue(db_path), conversation_store=conversations)

    V2Daemon(runtime)._migrate_blank_project_records()

    events = conversations.list_owner(owner_user_id="alex")
    assert [(event.content, event.project_id) for event in events] == [
        ("Unassociated", ""),
        ("Garden task", "garden"),
    ]


def test_startup_imports_legacy_user_and_project_ledgers_once_with_original_scope(tmp_path, monkeypatch) -> None:
    users = V2UserStore(":memory:")
    user = users.onboard(display_name="Alex", users_root=tmp_path / "profiles", user_id="alex")
    projects = ProjectStore(":memory:")
    project = projects.create_project(name="Garden", root_path=str(tmp_path / "garden"), owner_user_id=user.user_id)
    home = projects.ensure_home_project(user.user_id, root_path=str(tmp_path / "home"))
    conversations = SQLiteConversationStore(":memory:")
    runtime = build_runtime_host(user_store=users, project_store=projects, conversation_store=conversations)
    reads: list[str] = []

    def legacy(*, project_id: str, **_) -> str:
        reads.append(project_id)
        if project_id == home.project_id:
            return "## Migrated legacy Home ledger: alex/ledger-0001.md\n### Task one\n#### Conversation\n- User: Restore this turn\n#### Conversation\n- Alphonse: Restored reply."
        if project_id == project.project_id:
            return "### Task garden\n#### Conversation\n- User: Garden turn\n#### Conversation\n- Alphonse: Garden reply."
        return "### Task one\n#### Conversation\n- User: Restore this turn\n#### Conversation\n- Alphonse: Restored reply."

    monkeypatch.setattr(runtime.core.memory, "latest_content", legacy)
    daemon = V2Daemon(runtime)
    daemon._ensure_home_projects()
    daemon._ensure_home_projects()

    events = conversations.list_owner(owner_user_id="alex")
    assert [(event.role, event.project_id) for event in events[:2]] == [("user", ""), ("assistant", "")]
    assert sorted((event.role, event.project_id) for event in events[2:]) == sorted([
        ("user", project.project_id), ("assistant", project.project_id),
    ])
    assert [event.content for event in events] == ["Restore this turn", "Restored reply.", "Garden turn", "Garden reply."]
    assert reads.count("") == 1
    assert reads.count(home.project_id) == 1
    assert reads.count(project.project_id) == 1


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


def test_project_seen_cursor_is_authoritative_and_project_scoped() -> None:
    store = SQLiteConversationStore(":memory:")
    first = store.record(owner_user_id="alex", project_id="alpha", role="user", content="Alpha one", source="desktop", source_message_id="inbound:alpha-1")
    store.record(owner_user_id="alex", project_id="beta", role="assistant", content="Beta one", source="telegram", source_message_id="outbound:beta-1")

    assert store.project_has_unseen_messages(owner_user_id="alex", project_id="alpha")
    assert store.project_has_unseen_messages(owner_user_id="alex", project_id="beta")
    store.mark_project_seen(owner_user_id="alex", project_id="alpha", through_sequence=first.sequence)
    assert not store.project_has_unseen_messages(owner_user_id="alex", project_id="alpha")
    assert store.project_has_unseen_messages(owner_user_id="alex", project_id="beta")


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
    assert not store.project_has_unseen_messages(owner_user_id="alex", project_id="alpha")
    store.record(owner_user_id="alex", project_id="alpha", role="assistant", content="New", source="desktop", source_message_id="outbound:new")
    assert store.project_has_unseen_messages(owner_user_id="alex", project_id="alpha")


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


def test_desktop_history_imports_legacy_ledger_only_once(monkeypatch) -> None:
    runtime = build_runtime_host(conversation_store=SQLiteConversationStore(":memory:"))
    daemon = V2Daemon(runtime)
    project = runtime.project_store.create_project(name="Innovator", root_path="/tmp/innovator", owner_user_id="alex")
    reads = 0

    def ledger(**_) -> str:
        nonlocal reads
        reads += 1
        return "### Task one\n#### Conversation\n- User: Restore me\n#### Conversation\n- Alphonse: Restored."

    monkeypatch.setattr(runtime.core.memory, "latest_content", ledger)

    history = daemon.desktop_conversation_history(user="alex", project_id=project.project_id)
    repeated = daemon.desktop_conversation_history(user="alex", project_id=project.project_id)

    # Unknown historical dates must not be represented as today's messages.
    assert history == repeated == []
    imported = runtime.conversation_store.list_owner(owner_user_id="alex")
    assert [(item.role, item.content) for item in imported] == [("user", "Restore me"), ("assistant", "Restored.")]
    assert all(item.created_at == item.observed_at == "" for item in imported)
    assert all(item.provenance == "legacy_import" for item in imported)
    assert repeated == history
    assert reads == 1


def test_conversation_store_searches_memory_across_projects() -> None:
    store = SQLiteConversationStore(":memory:")
    store.record(
        owner_user_id="member-a", project_id="shopping", role="user",
        content="The pharmacy receipt total was 420 pesos.", source="test", source_message_id="memory-1",
    )
    store.record(
        owner_user_id="member-b", project_id="shopping", role="user",
        content="A separate account event.", source="test", source_message_id="memory-2",
    )

    matches = store.search(owner_user_id="member-a", query="pharmacy receipt", limit=5)

    assert [event.content for event in matches] == ["The pharmacy receipt total was 420 pesos."]
