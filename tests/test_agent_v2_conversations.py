from __future__ import annotations

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
