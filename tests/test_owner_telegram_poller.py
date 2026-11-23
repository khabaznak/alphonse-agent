from datetime import datetime, timezone

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_ingress import enqueue_telegram_owner
from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_poller import TelegramOwnerPoller
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.users import V2UserStore


def update(uid, timestamp, *, sender=123):
    return {'update_id': uid, 'message': {'date': timestamp, 'message_id': uid, 'text': f'Question {uid}',
        'from': {'id': sender}, 'chat': {'id': sender, 'type': 'private'}}}


class Client:
    def __init__(self, updates): self.updates, self.offsets = updates, []
    def get_updates(self, *, offset, timeout):
        self.offsets.append(offset)
        return [x for x in self.updates if offset is None or x['update_id'] >= offset]


def setup(tmp_path):
    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    users.bind_address(user_id=owner.user_id, integration_id='telegram-test', provider_key='telegram', provider_user_id='123')
    SQLiteConversationStore(path)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    with queue._connect() as conn:
        conn.execute('''CREATE TABLE v2_integrations (integration_id TEXT PRIMARY KEY, provider_key TEXT,
            enabled INTEGER, config_json TEXT, credential_generation INTEGER, revocation_pending INTEGER,
            credential_deletion_pending INTEGER, updated_at TEXT)''')
        conn.execute('INSERT INTO v2_integrations VALUES (?,?,?,?,?,?,?,?)', ('telegram-test','telegram',1,'{}',1,0,0,'r1'))
    return queue


def test_old_and_unmapped_updates_held_new_update_queued_once(tmp_path):
    queue = setup(tmp_path)
    client = Client([update(4, 1001, sender=999), update(1, 999), update(2, 1000), update(3, 1001)])
    poller = TelegramOwnerPoller(queue, 'telegram-test', client, activated_at=datetime.fromtimestamp(1000, timezone.utc))
    assert poller.poll_once() == {'queued': 1, 'held': 3, 'next_offset': 5}
    with queue._connect() as conn:
        assert conn.execute('SELECT count(*) FROM v2_inbound_messages').fetchone()[0] == 1
        assert [r['reason'] for r in conn.execute('SELECT reason FROM v2_owner_telegram_held_updates ORDER BY update_id')] == [
            'before_activation_or_unknown_time', 'before_activation_or_unknown_time', 'unsupported_or_unauthorized']
        assert conn.execute('SELECT update_json FROM v2_owner_telegram_held_updates WHERE update_id=4').fetchone()[0] == '{}'
    restarted = TelegramOwnerPoller(queue, 'telegram-test', client, activated_at=datetime.fromtimestamp(2000, timezone.utc))
    assert restarted.poll_once() == {'queued': 0, 'held': 0, 'next_offset': 5}
    assert client.offsets == [None, 5]


def test_replayed_update_after_enqueue_before_offset_does_not_duplicate(tmp_path):
    queue = setup(tmp_path)
    item = update(1, 1001)
    existing = enqueue_telegram_owner(queue, integration_id='telegram-test', update=item)
    poller = TelegramOwnerPoller(queue, 'telegram-test', Client([item]), activated_at=datetime.fromtimestamp(1000, timezone.utc))
    assert poller.poll_once()['next_offset'] == 2
    with queue._connect() as conn:
        assert conn.execute('SELECT message_id FROM v2_inbound_messages').fetchone()[0] == existing
        assert conn.execute('SELECT count(*) FROM v2_inbound_messages').fetchone()[0] == 1
