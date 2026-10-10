from __future__ import annotations

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.memory.daily_ledger import DailyLedgerProjector


def test_daily_ledger_projects_local_day_and_previous_day_summary(tmp_path) -> None:
    conversations = SQLiteConversationStore(":memory:")
    conversations.record(
        owner_user_id="member-a", project_id="", role="user",
        content="I shared the dinner receipt.", source="test", source_message_id="yesterday-user",
        created_at="2026-10-05T05:30:00+00:00",
    )
    conversations.record(
        owner_user_id="member-a", project_id="meal-plan", role="assistant",
        content="I recorded the total.", source="test", source_message_id="yesterday-assistant",
        created_at="2026-10-05T05:32:00+00:00",
    )
    projector = DailyLedgerProjector(
        conversation_store=conversations,
        users_root=lambda: tmp_path,
        timezone_provider=lambda: "America/Mexico_City",
        summarizer=lambda transcript: "Yesterday, the member shared a dinner receipt and its total was recorded.",
    )
    conversations.set_daily_ledger_projector(projector)

    conversations.record(
        owner_user_id="member-a", project_id="", role="user",
        content="What was the total?", source="test", source_message_id="today-user",
        created_at="2026-10-05T12:00:00+00:00",
    )

    previous = tmp_path / "member-a" / "memory" / "daily" / "2026-10-04.md"
    current = tmp_path / "member-a" / "memory" / "daily" / "2026-10-05.md"
    assert "I shared the dinner receipt." in previous.read_text(encoding="utf-8")
    assert "Yesterday, the member shared a dinner receipt" in current.read_text(encoding="utf-8")
    assert "What was the total?" in current.read_text(encoding="utf-8")


def test_daily_ledgers_are_isolated_by_member(tmp_path) -> None:
    conversations = SQLiteConversationStore(":memory:")
    projector = DailyLedgerProjector(
        conversation_store=conversations,
        users_root=lambda: tmp_path,
        timezone_provider=lambda: "UTC",
    )
    conversations.set_daily_ledger_projector(projector)

    conversations.record(
        owner_user_id="member-a", project_id="", role="user", content="Member A detail",
        source="test", source_message_id="member-a-event", created_at="2026-10-05T12:00:00+00:00",
    )
    conversations.record(
        owner_user_id="member-b", project_id="", role="user", content="Member B detail",
        source="test", source_message_id="member-b-event", created_at="2026-10-05T12:00:00+00:00",
    )

    root = tmp_path / "memory" / "daily"
    member_a = (tmp_path / "member-a" / root.relative_to(tmp_path) / "2026-10-05.md").read_text(encoding="utf-8")
    member_b = (tmp_path / "member-b" / root.relative_to(tmp_path) / "2026-10-05.md").read_text(encoding="utf-8")
    assert "Member A detail" in member_a and "Member B detail" not in member_a
    assert "Member B detail" in member_b and "Member A detail" not in member_b
