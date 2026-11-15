"""Local-owner review does not resend or expose another member's notifications."""
import os
import tempfile
from unittest.mock import patch

import pytest

from alphonse.agent_v2.core.io import ChannelAddress
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def test_normal_startup_runs_narrow_identity_notice_migration_after_lock(tmp_path, monkeypatch):
    db = tmp_path / 'fixture.sqlite'
    users = V2UserStore(db)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    outbox = SQLiteOutboundStore(db)
    text = 'Telegram message ignored because the sender is not mapped to an Alphonse user: provider_user_id=sender, chat_id=chat, reason=unmapped.'
    item = outbox.enqueue(address=ChannelAddress('tui', 'tui', owner.user_id, alphonse_user_id=owner.user_id),
        message=text, kind='identity_resolution', audience_user_id=owner.user_id,
        metadata={'provider_key': 'telegram', 'integration_id': 'fixture', 'provider_user_id': 'sender',
                  'channel_target': 'chat', 'reason': 'unmapped'})
    daemon = V2Daemon(build_runtime_host(user_store=users, outbox=outbox, schedule_store=ScheduledTaskStore(':memory:')))
    locked = []
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: locked.append(True))
    monkeypatch.setattr(daemon, 'stop', lambda: None)
    def after_migration():
        assert locked == [True]
        assert outbox.get(item.outbox_message_id).integration_id == 'desktop'
        raise RuntimeError('stop_fixture_startup')
    monkeypatch.setattr(daemon, '_ensure_home_projects', after_migration)
    with pytest.raises(RuntimeError, match='stop_fixture_startup'):
        daemon.start()


def test_peer_owner_can_review_and_archive_but_cannot_spoof_or_expose_another_member(tmp_path):
    db = tmp_path / 'fixture.sqlite'
    users = V2UserStore(db)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    member = users.create_user(display_name='Synthetic member')
    outbox = SQLiteOutboundStore(db)
    text = 'Telegram message ignored because the sender is not mapped to an Alphonse user: provider_user_id=sender, chat_id=chat, reason=unmapped.'
    own = outbox.enqueue(address=ChannelAddress('tui', 'tui', owner.user_id, alphonse_user_id=owner.user_id),
                          message=text, kind='identity_resolution', audience_user_id=owner.user_id,
                          metadata={'provider_key': 'telegram', 'integration_id': 'fixture', 'provider_user_id': 'sender',
                                    'channel_target': 'chat', 'reason': 'unmapped'})
    other = outbox.enqueue(address=ChannelAddress('tui', 'tui', member.user_id, alphonse_user_id=member.user_id),
                            message='Member private fixture', audience_user_id=member.user_id)
    protected = outbox.enqueue(address=ChannelAddress('tui', 'tui', owner.user_id, alphonse_user_id=owner.user_id),
                               message='Protected finding body', kind='finding', audience_user_id=owner.user_id)
    runtime = build_runtime_host(user_store=users, outbox=outbox, schedule_store=ScheduledTaskStore(':memory:'))
    daemon = V2Daemon(runtime)
    summary = daemon.poll_desktop(client_id='fixture', user=owner.user_id)['legacy_notification_review']
    assert summary == {'owner_scoped': 2, 'ambiguous': 0, 'unassigned': 0}
    with pytest.raises(PermissionError):
        daemon.poll_desktop(client_id='member-fixture', user=member.user_id)
    with tempfile.TemporaryDirectory(prefix='legacy-notice-', dir='/private/tmp') as socket_dir:
        server = V2DaemonServer(daemon, os.path.join(socket_dir, 'daemon.sock'))
        server.start()
        try:
            client = V2DaemonClient(server.socket_path)
            with pytest.raises(RuntimeError, match='legacy_notification_payload_denied'):
                client.request('legacy_notification_list', actor_user_id=member.user_id)
            items = client.request('legacy_notification_list')['items']
            by_id = {item['outbox_message_id']: item for item in items}
            assert set(by_id) == {own.outbox_message_id, protected.outbox_message_id}
            assert by_id[own.outbox_message_id]['message'] == text
            assert by_id[own.outbox_message_id]['delivery_certainty'] == 'not_attempted'
            assert by_id[protected.outbox_message_id]['message'] == ''
            assert by_id[protected.outbox_message_id]['can_archive'] is False
            assert by_id[protected.outbox_message_id]['review_restriction'] == 'linked_or_protected'
            with patch.object(server, '_local_peer_uid', side_effect=PermissionError('peer_denied')):
                with pytest.raises(RuntimeError, match='peer_denied'):
                    client.request('legacy_notification_list')
            with pytest.raises(RuntimeError, match='legacy_tui_recovery_unavailable'):
                client.request('legacy_notification_archive', message_id=other.outbox_message_id,
                               review_token=by_id[own.outbox_message_id]['review_token'])
            with pytest.raises(RuntimeError, match='legacy_tui_recovery_unavailable'):
                client.request('legacy_notification_archive', message_id=protected.outbox_message_id,
                               review_token=by_id[protected.outbox_message_id]['review_token'])
            result = client.request('legacy_notification_archive', message_id=own.outbox_message_id,
                                    review_token=by_id[own.outbox_message_id]['review_token'])
            assert result == {'status': 'archived_without_resending', 'delivery_certainty': 'not_attempted'}
            assert outbox.get(own.outbox_message_id).integration_id == 'retired_tui'
            assert outbox.get(other.outbox_message_id).integration_id == 'tui'
        finally:
            server.stop()
