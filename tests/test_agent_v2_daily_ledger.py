from __future__ import annotations

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.memory.daily_ledger import DailyLedgerProjector


@pytest.mark.skip(reason="daily projection suspended pending a cross-writer protocol")
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


@pytest.mark.skip(reason="daily projection suspended pending a cross-writer protocol")
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


def test_projection_is_unavailable_without_blocking_canonical_conversation(tmp_path):
    store = SQLiteConversationStore(tmp_path / 'history.sqlite')
    projector = DailyLedgerProjector(conversation_store=store, users_root=lambda: tmp_path / 'users')
    store.set_daily_ledger_projector(projector)
    event = store.record(owner_user_id='owner', role='user', content='synthetic', source='test',
                         source_message_id='fixture', created_at='2026-10-05T12:00:00+00:00')
    assert projector.available is False
    assert projector.refresh_for_event(event) is None
    with pytest.raises(RuntimeError, match='daily_ledger_projection_unavailable'):
        projector.rebuild_owner('owner')
    assert store.read(event_id=event.event_id, requester_user_id='owner').content == 'synthetic'
    assert not (tmp_path / 'users').exists()


@pytest.mark.parametrize('name', ['2026-10-05.md', 'household-notes.md', 'orphan.md.generated.sha256'])
def test_erasure_preserves_existing_daily_directory_and_event(tmp_path, name):
    store = SQLiteConversationStore(tmp_path / 'history.sqlite')
    projector = DailyLedgerProjector(conversation_store=store, users_root=lambda: tmp_path / 'users')
    store.set_daily_ledger_projector(projector)
    event = store.record(owner_user_id='owner', role='user', content='synthetic', source='test',
                         source_message_id='fixture', created_at='2026-10-05T12:00:00+00:00')
    root = tmp_path / 'users/owner/memory/daily'
    root.mkdir(parents=True)
    (root / name).write_text('Household-owned or ambiguous content')
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(ValueError, match='daily_ledger_reconciliation_required'):
        store.delete(event_id=event.event_id, owner_user_id='owner')
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before
    assert store.read(event_id=event.event_id, requester_user_id='owner') is not None


@pytest.mark.parametrize('component', ['owner', 'memory', 'daily'])
def test_invalidation_refuses_symlinked_scope_components(tmp_path, component):
    base = tmp_path / 'users'
    outside = tmp_path / 'outside'
    outside.mkdir()
    linked = {'owner': base / 'owner', 'memory': base / 'owner/memory',
              'daily': base / 'owner/memory/daily'}[component]
    linked.parent.mkdir(parents=True, exist_ok=True)
    linked.symlink_to(outside, target_is_directory=True)
    projector = DailyLedgerProjector(conversation_store=None, users_root=lambda: base)
    with pytest.raises(ValueError, match='daily_ledger_reconciliation_required'):
        projector.invalidate_owner('owner')
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize('owner', ['member/a', 'member a', '.member', 'member.', 'x' * 121])
def test_invalidation_refuses_owner_id_normalization_or_truncation(tmp_path, owner):
    projector = DailyLedgerProjector(conversation_store=None, users_root=lambda: tmp_path)
    with pytest.raises(ValueError, match='invalid_ledger_member'):
        projector.invalidate_owner(owner)
    assert list(tmp_path.iterdir()) == []
