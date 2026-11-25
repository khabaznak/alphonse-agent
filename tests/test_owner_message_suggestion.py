"""Advisory routing never submits work or turns uncertain judgments into suggestions."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from alphonse.agent_v2.owner_message_suggestion import suggest
from alphonse.agent_v2.system_one import SystemOneSettings


@pytest.mark.parametrize(('update', 'followup', 'expected'), [
    (.94, .08, 'steer'), (.04, .95, 'queue'), (.5, .5, None),
    (.95, .9, None), (.9, .4, None), (True, .1, None), (float('nan'), .1, None),
])
def test_bounded_judgments(update, followup, expected):
    client = Mock()
    client.evaluate.return_value = {'answers': {
        'updates_active': {'type': 'noul', 'noul': update},
        'separate_followup': {'type': 'noul', 'noul': followup},
    }}
    settings = SystemOneSettings(enabled=True, api_key='test', validated_at='test')
    assert suggest(settings, active_request='Find a recipe', draft='Use pears', client=client) == expected
    assert client.evaluate.call_args.kwargs['state'] == {'active_request': 'Find a recipe', 'draft': 'Use pears'}
    assert client.evaluate.call_args.kwargs['retries'] == 0


def test_provider_errors_and_disabled_setting_return_no_suggestion():
    client = Mock()
    client.evaluate.side_effect = RuntimeError('unavailable')
    settings = SystemOneSettings(enabled=True, api_key='test', validated_at='test')
    assert suggest(settings, active_request='Find a recipe', draft='Use pears', client=client) is None
    client.reset_mock()
    assert suggest(SystemOneSettings(), active_request='Find a recipe', draft='Use pears', client=client) is None
    client.evaluate.assert_not_called()


def test_ipc_only_suggests_for_owned_live_request(tmp_path, monkeypatch):
    import os
    from alphonse.agent_v2.ipc import PROTOCOL_VERSION, V2DaemonServer
    from alphonse.agent_v2.users import V2UserStore
    from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
    from alphonse.agent_v2.conversations import SQLiteConversationStore
    import alphonse.agent_v2.owner_message_suggestion as module

    db = tmp_path / 'fixture.db'
    users = V2UserStore(db)
    owner = users.onboard(display_name='Fixture', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(db, experimental_claim_fencing=True)
    SQLiteConversationStore(db)
    message_id = queue.enqueue_trusted_local_owner(prompt='Find a recipe', peer_uid=os.getuid())
    daemon = SimpleNamespace(experimental_owner_only=True, runtime=SimpleNamespace(user_store=users, queue=queue,
        system_one_settings_store=SimpleNamespace(get=lambda: SystemOneSettings())), owner_message_status=Mock(return_value='pending'))
    server = V2DaemonServer(daemon, experimental_owner_ingress=True)
    monkeypatch.setattr(server, '_local_peer_uid', lambda connection: os.getuid())
    judge = Mock(return_value='steer')
    monkeypatch.setattr(module, 'suggest', judge)
    req = {'version': PROTOCOL_VERSION, 'params': {'active_message_id': message_id, 'prompt': 'Use pears'}}
    assert server._owner_message_suggestion(None, req) == {'suggestion': 'steer'}
    assert judge.call_args.kwargs == {'active_request': 'Find a recipe', 'draft': 'Use pears'}
    daemon.owner_message_status.side_effect = ['pending', 'ready']
    assert server._owner_message_suggestion(None, req) == {'suggestion': None}
    daemon.owner_message_status.side_effect = None
    daemon.owner_message_status.return_value = 'pending'
    monkeypatch.setattr(server, '_local_peer_uid', lambda connection: os.getuid() + 1)
    assert server._owner_message_suggestion(None, req) == {'suggestion': None}
