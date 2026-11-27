from dataclasses import replace
from datetime import datetime, timezone
import threading

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_listener import TelegramOwnerListener
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.integrations.store import IntegrationConfigRecord
from alphonse.agent_v2.users import V2UserStore


class Store:
    def __init__(self, path):
        self.db_path = str(path)
        self.record = IntegrationConfigRecord('telegram-test', 'telegram', 'Telegram', True, updated_at='v1')
    def get(self, _): return self.record
    def resolve_credentials(self, _): return {'bot_token': 'synthetic'}


def test_explicit_start_uses_poller_and_stops_without_legacy_outbox(tmp_path):
    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    users.bind_address(user_id=owner.user_id, integration_id='telegram-test', provider_key='telegram', provider_user_id='123')
    SQLiteConversationStore(path)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    with queue._connect() as conn:
        conn.execute('''CREATE TABLE v2_integrations (integration_id TEXT PRIMARY KEY,provider_key TEXT,enabled INTEGER,
            config_json TEXT,credential_generation INTEGER,revocation_pending INTEGER,credential_deletion_pending INTEGER,updated_at TEXT)''')
        conn.execute('INSERT INTO v2_integrations VALUES (?,?,?,?,?,?,?,?)', ('telegram-test','telegram',1,'{}',0,0,0,'v1'))
    visited = threading.Event()
    sent = []
    class Client:
        def get_updates(self, *, offset, timeout):
            visited.set()
            return []
        def send_message(self, *, chat_id, text):
            sent.append((chat_id, text))
            return 'provider-id'
    store = Store(path)
    listener = TelegramOwnerListener(queue, store, 'telegram-test', client_factory=lambda **_: Client(),
        clock=lambda: datetime.fromtimestamp(1000, timezone.utc), poll_interval=60)
    with queue._connect() as conn:
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='v2_owner_telegram_poll_state'").fetchone()
    listener.start()
    try:
        assert visited.wait(2)
        assert listener.send_reply(integration_id='telegram-test', chat_id='123', text='Synthetic reply') == 'provider-id'
        assert sent == [('123', 'Synthetic reply')]
        with pytest.raises(PermissionError):
            listener.send_reply(integration_id='other', chat_id='123', text='Synthetic reply')
        store.record = replace(store.record, updated_at='v2')
        with pytest.raises(PermissionError):
            listener.send_reply(integration_id='telegram-test', chat_id='123', text='Synthetic reply')
        store.record = replace(store.record, updated_at='v1')
        with queue._connect() as conn:
            assert conn.execute('SELECT activated_at_epoch FROM v2_owner_telegram_poll_state').fetchone()[0] == 1000
            assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='v2_outbound_messages'").fetchone()
    finally:
        assert listener.stop(timeout=2)
    with pytest.raises(PermissionError):
        listener.send_reply(integration_id='telegram-test', chat_id='123', text='Synthetic reply')


def test_disabled_or_changed_config_cannot_poll(tmp_path):
    path = tmp_path / 'owner.sqlite'
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    store = Store(path)
    store.record = replace(store.record, enabled=False)
    listener = TelegramOwnerListener(queue, store, 'telegram-test', client_factory=lambda **_: object())
    with pytest.raises(PermissionError):
        listener.start()


def test_feedback_refresh_failure_and_receipt_deduplication(tmp_path):
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_ingress import enqueue_telegram_owner
    import time

    path = tmp_path / 'feedback.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    users.bind_address(user_id=owner.user_id, integration_id='telegram-test', provider_key='telegram', provider_user_id='123')
    SQLiteConversationStore(path)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    with queue._connect() as conn:
        conn.execute('''CREATE TABLE v2_integrations (integration_id TEXT PRIMARY KEY,provider_key TEXT,enabled INTEGER,
            config_json TEXT,credential_generation INTEGER,revocation_pending INTEGER,credential_deletion_pending INTEGER,updated_at TEXT)''')
        conn.execute('INSERT INTO v2_integrations VALUES (?,?,?,?,?,?,?,?)', ('telegram-test','telegram',1,'{}',0,0,0,'v1'))
    class Client:
        def __init__(self): self.events = []
        def get_updates(self, *, offset, timeout): return []
        def set_message_reaction(self, **kwargs):
            self.events.append(('receipt', kwargs['message_id'], kwargs['emoji']))
            raise RuntimeError('provider failure')
        def send_chat_action(self, **kwargs):
            self.events.append(('activity', kwargs['action']))
    client = Client()
    listener = TelegramOwnerListener(queue, Store(path), 'telegram-test', client_factory=lambda **_: client,
                                     poll_interval=60, activity_interval=.1)
    listener.start()
    try:
        update = {'update_id': 1, 'message': {'message_id': 41, 'text': 'Hello', 'from': {'id': 123},
                                               'chat': {'id': 123, 'type': 'private'}}}
        mid = enqueue_telegram_owner(queue, integration_id='telegram-test', update=update)
        listener._queue_receipt(mid)
        listener._queue_receipt(mid)
        deadline = time.monotonic() + 3
        while len(client.events) < 1 and time.monotonic() < deadline: time.sleep(.01)
        assert client.events == [('receipt', '41', '👀')]
        second_update = {'update_id': 2, 'message': {'message_id': 42, 'text': 'Next', 'from': {'id': 123},
                                                      'chat': {'id': 123, 'type': 'private'}}}
        second_mid = enqueue_telegram_owner(queue, integration_id='telegram-test', update=second_update)
        listener._queue_receipt(second_mid)
        cancelled = threading.Event()
        activity = listener.begin_activity(mid, cancelled=cancelled.is_set)
        deadline = time.monotonic() + 3
        while len(client.events) < 3 and time.monotonic() < deadline: time.sleep(.01)
        assert len(client.events) >= 3
        assert client.events[1:] == [('activity', 'typing')] * (len(client.events) - 1)
        cancelled.set()
        time.sleep(.25)
        count = len(client.events)
        time.sleep(.2)
        assert len(client.events) == count
        assert [event for event in client.events if event[0] == 'receipt'] == [('receipt', '41', '👀')]
        activity.set()
        with queue._connect() as conn:
            assert conn.execute('SELECT count(*) FROM v2_owner_telegram_receipts').fetchone()[0] == 2
    finally:
        listener.stop(timeout=2)


def test_receipt_scan_recovers_pending_without_wakeups_or_replay(tmp_path):
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_ingress import enqueue_telegram_owner
    import time

    path = tmp_path / 'overflow.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    users.bind_address(user_id=owner.user_id, integration_id='telegram-test', provider_key='telegram', provider_user_id='123')
    SQLiteConversationStore(path)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    with queue._connect() as conn:
        conn.execute('''CREATE TABLE v2_integrations (integration_id TEXT PRIMARY KEY,provider_key TEXT,enabled INTEGER,
            config_json TEXT,credential_generation INTEGER,revocation_pending INTEGER,credential_deletion_pending INTEGER,updated_at TEXT)''')
        conn.execute('INSERT INTO v2_integrations VALUES (?,?,?,?,?,?,?,?)', ('telegram-test','telegram',1,'{}',0,0,0,'v1'))
    started, release = threading.Event(), threading.Event()
    events = []
    class Client:
        def get_updates(self, *, offset, timeout): return []
        def set_message_reaction(self, **kwargs):
            events.append(kwargs['message_id'])
            started.set()
            release.wait(5)
    listener = TelegramOwnerListener(queue, Store(path), 'telegram-test', client_factory=lambda **_: Client(), poll_interval=60)
    real_origin = listener._origin_locked
    claim_failures = []
    def transient_claim_failure(*args, **kwargs):
        if not claim_failures:
            claim_failures.append(True)
            raise RuntimeError('transient storage failure')
        return real_origin(*args, **kwargs)
    listener._origin_locked = transient_claim_failure
    listener.start()
    try:
        # Do not issue wakeups: durable pending rows must be scanned anyway.
        for number in (41, 42):
            mid = enqueue_telegram_owner(queue, integration_id='telegram-test', update={
                'update_id': number, 'message': {'message_id': number, 'text': 'Hello',
                'from': {'id': 123}, 'chat': {'id': 123, 'type': 'private'}}})
        assert started.wait(3)
        # Restart while the first provider call is unresolved. Its committed
        # attempt must not replay; the other pending receipt must be recovered.
        assert listener.stop(timeout=2)
        listener.start()
        release.set()
        deadline = time.monotonic() + 4
        while len(events) < 2 and time.monotonic() < deadline: time.sleep(.01)
        assert sorted(events) == ['41', '42']
        assert claim_failures == [True]
        with queue._connect() as conn:
            assert [row['status'] for row in conn.execute('SELECT status FROM v2_owner_telegram_receipts')] == ['attempted', 'attempted']
    finally:
        release.set()
        listener.stop(timeout=2)


def test_old_receipt_markers_migrate_without_replay(tmp_path):
    queue = SQLiteMessageQueue(tmp_path / 'old.sqlite', experimental_claim_fencing=True)
    with queue._connect() as conn:
        conn.execute('CREATE TABLE v2_owner_telegram_receipts (message_id TEXT PRIMARY KEY) STRICT')
        conn.execute("INSERT INTO v2_owner_telegram_receipts VALUES ('old')")
        TelegramOwnerListener._receipt_schema(conn)
        assert conn.execute("SELECT status FROM v2_owner_telegram_receipts WHERE message_id='old'").fetchone()[0] == 'attempted'


def test_revoked_receipt_is_retired_without_provider_call(tmp_path):
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_ingress import enqueue_telegram_owner
    import time
    path = tmp_path / 'revoked.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    users.bind_address(user_id=owner.user_id, integration_id='telegram-test', provider_key='telegram', provider_user_id='123')
    SQLiteConversationStore(path)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    with queue._connect() as conn:
        conn.execute('''CREATE TABLE v2_integrations (integration_id TEXT PRIMARY KEY,provider_key TEXT,enabled INTEGER,
            config_json TEXT,credential_generation INTEGER,revocation_pending INTEGER,credential_deletion_pending INTEGER,updated_at TEXT)''')
        conn.execute('INSERT INTO v2_integrations VALUES (?,?,?,?,?,?,?,?)', ('telegram-test','telegram',1,'{}',0,0,0,'v1'))
    mid = enqueue_telegram_owner(queue, integration_id='telegram-test', update={'update_id': 1, 'message': {
        'message_id': 41, 'text': 'Hello', 'from': {'id': 123}, 'chat': {'id': 123, 'type': 'private'}}})
    with queue._connect() as conn:
        conn.execute('UPDATE v2_user_addresses SET is_active=0')
    events = []
    class Client:
        def get_updates(self, *, offset, timeout): return []
        def set_message_reaction(self, **kwargs): events.append(kwargs)
    listener = TelegramOwnerListener(queue, Store(path), 'telegram-test', client_factory=lambda **_: Client(), poll_interval=60)
    listener.start()
    try:
        deadline = time.monotonic() + 3
        status = None
        while time.monotonic() < deadline:
            with queue._connect() as conn:
                status = conn.execute('SELECT status FROM v2_owner_telegram_receipts WHERE message_id=?', (mid,)).fetchone()[0]
            if status == 'discarded': break
            time.sleep(.01)
        assert status == 'discarded'
        assert events == []
    finally:
        listener.stop(timeout=2)
