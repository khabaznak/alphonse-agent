"""Preserve owner identity notices as the terminal interface is retired."""
from types import SimpleNamespace
from datetime import datetime, timezone

import pytest
from alphonse.agent_v2.core.io.channels import ChannelAddress, channel_metadata
from alphonse.agent_v2.core.io.identity import V2IdentityResolver
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.messages.channel import CommunicationChannel
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore, OutboundSelector
from alphonse.agent_v2.integrations.discord.runtime import DiscordIntegrationRuntime, DiscordRuntimeStats
from alphonse.agent_v2.integrations.telegram.runtime import TelegramIntegrationRuntime
from alphonse.agent_v2.users import V2UserStore
from alphonse.agent_v2.retention import prune_operational_data


def test_local_channel_defaults_to_desktop_but_explicit_legacy_identity_survives():
    channel = CommunicationChannel(InMemoryMessageQueue())
    current = channel.queue_message(prompt='Hello', user='owner')
    assert current.message.metadata['channel']['integration_id'] == 'desktop'
    assert current.message.metadata['channel']['provider_key'] == 'tui'
    assert channel_metadata(integration_id='', provider_key='')['integration_id'] == 'desktop'
    assert channel_metadata(integration_id='', provider_key='')['provider_key'] == 'tui'
    assert V2IdentityResolver().integration_for().integration_id == 'desktop'
    old = channel.queue_message(prompt='Old route', user='owner', integration_id='tui')
    assert old.message.metadata['channel']['integration_id'] == 'tui'


def test_new_unknown_sender_notices_go_to_owner_desktop(tmp_path):
    store = SQLiteOutboundStore(tmp_path / 'outbox.db')
    telegram = SimpleNamespace(outbox=store, owner_user_id='owner', integration_id='telegram-one')
    TelegramIntegrationRuntime._notify_owner_unresolved_user(
        telegram, provider_user_id='unknown-tg', chat_id='chat', reason='unmapped', update={})
    discord = SimpleNamespace(outbox=store, owner_user_id='owner', integration_id='discord-one',
                              _stats=DiscordRuntimeStats(), access_request_store=None)
    DiscordIntegrationRuntime._handle_unknown_user(
        discord, raw={}, channel_id='channel', guild_id='guild', provider_user_id='unknown-discord')
    deliveries = [store.claim_next(OutboundSelector(integration_id='desktop', channel_target='owner'),
                                   lease_owner='desktop:test') for _ in range(2)]
    assert all(item and item.kind == 'identity_resolution' and item.audience_user_id == 'owner'
               and item.provider_key == 'tui' for item in deliveries)
    assert {item.metadata['provider_key'] for item in deliveries} == {'telegram', 'discord'}


def test_only_canonical_owners_untouched_identity_notices_move(tmp_path):
    db = tmp_path / 'household.db'
    users = V2UserStore(db)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    member = users.create_user(display_name='Synthetic member')
    store = SQLiteOutboundStore(db)

    def notice(user, provider='telegram', *, kind='identity_resolution'):
        metadata = {'provider_key': provider, 'integration_id': 'fixture',
                    'provider_user_id': 'fixture-user', 'channel_target': 'fixture-chat', 'reason': 'unmapped'}
        content = ('Telegram message ignored because the sender is not mapped to an Alphonse user: '
                   'provider_user_id=fixture-user, chat_id=fixture-chat, reason=unmapped.')
        if provider == 'discord':
            content = ('Discord message ignored because the sender is not mapped to an Alphonse user: '
                       'provider_user_id=fixture-user, channel_id=fixture-chat.')
        return store.enqueue(address=ChannelAddress('tui', 'tui', user, alphonse_user_id=user),
            message=content, kind=kind, audience_user_id=user, metadata=metadata)

    eligible = notice(owner.user_id)
    member_notice = notice(member.user_id)
    attempted = notice(owner.user_id, 'discord')
    other_kind = notice(owner.user_id, kind='response')
    unsupported = notice(owner.user_id, 'unknown')
    protected = store.enqueue(address=ChannelAddress('tui', 'tui', owner.user_id, alphonse_user_id=owner.user_id),
        message='Linked notice', kind='identity_resolution', audience_user_id=owner.user_id,
        metadata={'provider_key': 'telegram', 'integration_id': 'fixture', 'occurrence_key': 'scheduled'})
    with store._connect() as conn:
        conn.execute('UPDATE v2_outbox SET attempt_count=1 WHERE outbox_message_id=?', (attempted.outbox_message_id,))
    assert store.migrate_untouched_tui_identity_notices(member.user_id) == 0
    assert store.migrate_untouched_tui_identity_notices(owner.user_id) == 1
    migrated = store.get(eligible.outbox_message_id)
    assert migrated.integration_id == 'desktop' and migrated.message == eligible.message
    assert migrated.audience_user_id == owner.user_id and migrated.outbox_message_id == eligible.outbox_message_id
    assert store.migrate_untouched_tui_identity_notices(owner.user_id) == 0
    for original in (member_notice, attempted, other_kind, unsupported, protected):
        assert store.get(original.outbox_message_id).integration_id == 'tui'


def test_missing_or_inactive_authority_cannot_move_notices(tmp_path):
    isolated = SQLiteOutboundStore(tmp_path / 'isolated.db')
    assert isolated.migrate_untouched_tui_identity_notices('owner') == 0
    db = tmp_path / 'household.db'
    users = V2UserStore(db)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    store = SQLiteOutboundStore(db)
    notice = store.enqueue(address=ChannelAddress('tui', 'tui', owner.user_id, alphonse_user_id=owner.user_id),
        message='Synthetic notice', kind='identity_resolution', audience_user_id=owner.user_id,
        metadata={'provider_key': 'telegram', 'integration_id': 'fixture'})
    users.update_user(owner.user_id, is_active=False)
    assert store.migrate_untouched_tui_identity_notices(owner.user_id) == 0
    assert store.get(notice.outbox_message_id).integration_id == 'tui'


def test_legacy_summary_exposes_only_counts_to_active_admin(tmp_path):
    db = tmp_path / 'household.db'
    users = V2UserStore(db)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    member = users.create_user(display_name='Synthetic member')
    store = SQLiteOutboundStore(db)
    def message(audience, target):
        return store.enqueue(address=ChannelAddress('tui', 'tui', target, alphonse_user_id=audience),
                             message='Never return this content', audience_user_id=audience)
    owned = message(owner.user_id, owner.user_id)
    ambiguous = message(owner.user_id, member.user_id)
    member_item = message(member.user_id, member.user_id)
    unassigned = message(owner.user_id, owner.user_id)
    delivered = message(owner.user_id, owner.user_id)
    with store._connect() as conn:
        conn.execute("UPDATE v2_outbox SET audience_user_id='',channel_target='' WHERE outbox_message_id=?",
                     (unassigned.outbox_message_id,))
        conn.execute("UPDATE v2_outbox SET status='delivered' WHERE outbox_message_id=?", (delivered.outbox_message_id,))
    assert store.legacy_tui_unresolved_summary(owner.user_id) == {'owner_scoped': 1, 'ambiguous': 1, 'unassigned': 1}
    assert store.legacy_tui_unresolved_summary(member.user_id) is None
    assert store.get(owned.outbox_message_id).message == 'Never return this content'
    assert store.get(ambiguous.outbox_message_id).integration_id == 'tui'
    assert store.get(member_item.outbox_message_id).integration_id == 'tui'
    users.update_user(owner.user_id, is_active=False)
    assert store.legacy_tui_unresolved_summary(owner.user_id) is None


def test_owner_can_review_and_archive_without_resend_or_false_delivery(tmp_path):
    db = tmp_path / 'household.db'
    users = V2UserStore(db)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    member = users.create_user(display_name='Synthetic member')
    store = SQLiteOutboundStore(db)
    def message(who, *, status='pending'):
        item = store.enqueue(address=ChannelAddress('tui', 'tui', who, alphonse_user_id=who),
            message='Telegram message ignored because the sender is not mapped to an Alphonse user: provider_user_id=x, chat_id=y, reason=z.',
            audience_user_id=who, kind='identity_resolution', metadata={'provider_key':'telegram',
            'integration_id':'fixture', 'provider_user_id':'x', 'channel_target':'y', 'reason':'z'})
        with store._connect() as conn:
            conn.execute('UPDATE v2_outbox SET status=? WHERE outbox_message_id=?', (status, item.outbox_message_id))
        return item
    own = message(owner.user_id)
    retried = message(owner.user_id, status='retry_wait')
    other = message(member.user_id)
    with store._connect() as conn:
        conn.execute("UPDATE v2_outbox SET attempt_count=1 WHERE outbox_message_id=?", (retried.outbox_message_id,))
    items = store.list_legacy_tui_owner_items(owner.user_id)
    assert {item['outbox_message_id'] for item in items} == {own.outbox_message_id, retried.outbox_message_id}
    assert all(item['message'].startswith('Telegram message ignored') for item in items)
    with pytest.raises(ValueError, match='legacy_tui_recovery_unavailable'):
        store.list_legacy_tui_owner_items(member.user_id)
    by_id = {item['outbox_message_id']: item for item in items}
    assert by_id[own.outbox_message_id]['delivery_certainty'] == 'not_attempted'
    assert by_id[retried.outbox_message_id]['delivery_certainty'] == 'unknown'
    outcome = store.archive_legacy_tui_owner_item(owner.user_id, own.outbox_message_id, by_id[own.outbox_message_id]['review_token'])
    assert outcome == {'status': 'archived_without_resending', 'delivery_certainty': 'not_attempted'}
    archived = store.get(own.outbox_message_id)
    assert archived.integration_id == 'retired_tui' and archived.status == 'pending'
    assert archived.message.startswith('Telegram message ignored') and archived.audience_user_id == owner.user_id
    assert archived.metadata['legacy_tui_retirement']['action'] == 'owner_archived_without_resending'
    assert store.claim_next(OutboundSelector(integration_id='retired_tui')) is None
    assert own.outbox_message_id not in {row.outbox_message_id for row in store.list_pending()}
    assert [row.outbox_message_id for row in store.list(OutboundSelector(integration_id='retired_tui'))] == [own.outbox_message_id]
    assert not store.mark_delivered(own.outbox_message_id, provider_message_id='late')
    assert not store.mark_failed(own.outbox_message_id, error='late')
    assert store.get(own.outbox_message_id).status == 'pending'
    with pytest.raises(ValueError, match='legacy_tui_recovery_unavailable'):
        store.archive_legacy_tui_owner_item(owner.user_id, other.outbox_message_id, by_id[own.outbox_message_id]['review_token'])
    assert store.get(other.outbox_message_id).integration_id == 'tui'


def test_legacy_review_rejects_stale_changed_claimed_and_revoked_rows(tmp_path):
    db = tmp_path / 'household.db'
    users = V2UserStore(db)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    store = SQLiteOutboundStore(db)
    item = store.enqueue(address=ChannelAddress('tui', 'tui', owner.user_id, alphonse_user_id=owner.user_id),
        message='Telegram message ignored because the sender is not mapped to an Alphonse user: provider_user_id=x, chat_id=y, reason=z.',
        audience_user_id=owner.user_id, kind='identity_resolution', metadata={'provider_key':'telegram',
        'integration_id':'fixture', 'provider_user_id':'x', 'channel_target':'y', 'reason':'z'})
    old = store.list_legacy_tui_owner_items(owner.user_id)[0]
    with store._connect() as conn:
        conn.execute("UPDATE v2_outbox SET updated_at='2000-01-01T00:00:00+00:00' WHERE outbox_message_id=?", (item.outbox_message_id,))
    with pytest.raises(ValueError, match='legacy_tui_recovery_unavailable'):
        store.archive_legacy_tui_owner_item(owner.user_id, item.outbox_message_id, old['review_token'])
    changed = store.list_legacy_tui_owner_items(owner.user_id)[0]
    with store._connect() as conn:
        conn.execute("UPDATE v2_outbox SET status='claimed',lease_owner='old-worker' WHERE outbox_message_id=?", (item.outbox_message_id,))
    claimed = store.list_legacy_tui_owner_items(owner.user_id)[0]
    assert claimed['can_archive'] is False
    with pytest.raises(ValueError, match='legacy_tui_recovery_unavailable'):
        store.archive_legacy_tui_owner_item(owner.user_id, item.outbox_message_id, claimed['review_token'])
    with store._connect() as conn:
        conn.execute("UPDATE v2_outbox SET status='pending',lease_owner='' WHERE outbox_message_id=?", (item.outbox_message_id,))
    current = store.list_legacy_tui_owner_items(owner.user_id)[0]
    users.update_user(owner.user_id, is_active=False)
    with pytest.raises(ValueError, match='legacy_tui_recovery_unavailable'):
        store.archive_legacy_tui_owner_item(owner.user_id, item.outbox_message_id, current['review_token'])
    assert store.get(item.outbox_message_id).integration_id == 'tui'


def test_protected_finding_scheduled_and_question_rows_are_redacted_and_unmodified(tmp_path):
    db = tmp_path / 'household.db'
    users = V2UserStore(db)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    store = SQLiteOutboundStore(db)
    protected = []
    for kind, metadata, extra in [
        ('finding', {'finding_id': 'revoked'}, {}),
        ('response', {'occurrence_key': 'scheduled'}, {}),
        ('question', {}, {'question_id': 'q1'}),
        ('response', {}, {'task_id': 'task1'}),
        ('identity_resolution', {'provider_key':'telegram','integration_id':'fixture',
          'provider_user_id':'x','channel_target':'y','reason':'z'}, {}),
    ]:
        protected.append(store.enqueue(address=ChannelAddress('tui', 'tui', owner.user_id, alphonse_user_id=owner.user_id),
            message='Protected private content', kind=kind, audience_user_id=owner.user_id, metadata=metadata, **extra))
    rows = store.list_legacy_tui_owner_items(owner.user_id)
    assert len(rows) == len(protected)
    for row in rows:
        assert row['message'] == '' and row['can_archive'] is False
        assert row['review_restriction'] == 'linked_or_protected'
        with pytest.raises(ValueError, match='legacy_tui_recovery_unavailable'):
            store.archive_legacy_tui_owner_item(owner.user_id, row['outbox_message_id'], row['review_token'])
    assert all(store.get(item.outbox_message_id).integration_id == 'tui' for item in protected)


def test_owner_may_archive_unlinked_opaque_response_without_exposing_or_resending(tmp_path):
    db = tmp_path / 'household.db'
    users = V2UserStore(db)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    store = SQLiteOutboundStore(db)
    item = store.enqueue(address=ChannelAddress('tui', 'tui', owner.user_id, alphonse_user_id=owner.user_id),
        message='Synthetic private text', kind='response', audience_user_id=owner.user_id)
    row = store.list_legacy_tui_owner_items(owner.user_id)[0]
    assert row['review_restriction'] == 'opaque_response' and row['message'] == ''
    assert row['can_archive'] is True and row['delivery_certainty'] == 'not_attempted'
    result = store.archive_legacy_tui_owner_item(owner.user_id, item.outbox_message_id, row['review_token'])
    assert result == {'status': 'archived_without_resending', 'delivery_certainty': 'not_attempted'}
    saved = store.get(item.outbox_message_id)
    assert saved.message == 'Synthetic private text' and saved.integration_id == 'retired_tui'
    assert saved.metadata['legacy_tui_retirement']['action'] == 'owner_archived_without_resending'
    assert store.claim_next(OutboundSelector(integration_id='retired_tui')) is None


def test_opaque_response_rechecks_claim_and_source_metadata_before_archival(tmp_path):
    db = tmp_path / 'household.db'
    users = V2UserStore(db)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    store = SQLiteOutboundStore(db)
    item = store.enqueue(address=ChannelAddress('tui', 'tui', owner.user_id, alphonse_user_id=owner.user_id),
        message='Synthetic private text', kind='response', audience_user_id=owner.user_id)
    before = store.list_legacy_tui_owner_items(owner.user_id)[0]
    with store._connect() as conn:
        conn.execute("UPDATE v2_outbox SET metadata_json=? WHERE outbox_message_id=?", ('{\"finding_id\":\"private\"}', item.outbox_message_id))
    with pytest.raises(ValueError, match='legacy_tui_recovery_unavailable'):
        store.archive_legacy_tui_owner_item(owner.user_id, item.outbox_message_id, before['review_token'])
    protected = store.list_legacy_tui_owner_items(owner.user_id)[0]
    assert protected['message'] == '' and not protected['can_archive']
    assert protected['review_restriction'] == 'linked_or_protected'


def test_opaque_response_rejects_other_owner_and_active_claim(tmp_path):
    db = tmp_path / 'household.db'
    users = V2UserStore(db)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    member = users.create_user(display_name='Synthetic member')
    store = SQLiteOutboundStore(db)
    item = store.enqueue(address=ChannelAddress('tui', 'tui', owner.user_id, alphonse_user_id=owner.user_id),
        message='Synthetic private text', kind='response', audience_user_id=owner.user_id)
    before = store.list_legacy_tui_owner_items(owner.user_id)[0]
    with pytest.raises(ValueError, match='legacy_tui_recovery_unavailable'):
        store.archive_legacy_tui_owner_item(member.user_id, item.outbox_message_id, before['review_token'])
    with store._connect() as conn:
        conn.execute("UPDATE v2_outbox SET status='claimed',lease_owner='fixture' WHERE outbox_message_id=?", (item.outbox_message_id,))
    with pytest.raises(ValueError, match='legacy_tui_recovery_unavailable'):
        store.archive_legacy_tui_owner_item(owner.user_id, item.outbox_message_id, before['review_token'])
    assert store.get(item.outbox_message_id).integration_id == 'tui'


def test_failed_unresolved_and_archived_legacy_rows_survive_operational_pruning(tmp_path):
    db = tmp_path / 'household.db'
    users = V2UserStore(db)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    store = SQLiteOutboundStore(db)
    def old_notice(route):
        item = store.enqueue(address=ChannelAddress(route, 'tui', owner.user_id, alphonse_user_id=owner.user_id),
                             message='Synthetic notice', audience_user_id=owner.user_id)
        with store._connect() as conn:
            conn.execute("UPDATE v2_outbox SET status='failed',updated_at='2000-01-01T00:00:00+00:00' WHERE outbox_message_id=?", (item.outbox_message_id,))
        return item
    unresolved = old_notice('tui')
    archived = old_notice('tui')
    ordinary = old_notice('desktop')
    with store._connect() as conn:
        conn.execute("UPDATE v2_outbox SET integration_id='retired_tui' WHERE outbox_message_id=?", (archived.outbox_message_id,))
    with pytest.raises(ValueError, match='outbound_integration_id_reserved'):
        old_notice('retired_tui')
    prune_operational_data(db, now=datetime(2026, 11, 14, tzinfo=timezone.utc), retention_days=30)
    assert store.get(unresolved.outbox_message_id) is not None
    assert store.get(archived.outbox_message_id) is not None
    assert store.get(ordinary.outbox_message_id) is None
