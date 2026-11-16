"""New owner Desktop reminders use disposable databases and a real local IPC socket."""
import os
import socket
import tempfile
import threading
from datetime import datetime, timedelta, timezone

import pytest

from alphonse.agent_v2.owner_desktop_reminders import OwnerDesktopReminders
from alphonse.agent_v2.users import V2UserStore


def _fixture(tmp_path):
    path = str(tmp_path / 'fixture.sqlite')
    users = V2UserStore(path)
    owner = users.onboard(display_name='Fixture owner', users_root=tmp_path / 'users')
    return path, owner.user_id, OwnerDesktopReminders(path)


def test_reminder_atomic_offer_ack_and_restart_uncertainty(tmp_path):
    path, owner, reminders = _fixture(tmp_path)
    now = datetime.now(timezone.utc)
    due = now + timedelta(minutes=1)
    created = reminders.create(owner, 'Fixture reminder', due.isoformat(), 'a' * 32, now=now)
    assert reminders.create(owner, 'Fixture reminder', due.isoformat(), 'a' * 32, now=due) == created
    with pytest.raises(ValueError, match='conflict'):
        reminders.create(owner, 'Changed reminder', due.isoformat(), 'a' * 32, now=now)
    assert reminders.take_due(owner, now=now) == {'reminder': None}
    results = []
    threads = [threading.Thread(target=lambda: results.append(reminders.take_due(owner, now=due))) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    offered = [item['reminder'] for item in results if item['reminder'] is not None]
    assert len(offered) == 1
    reminder = offered[0]
    restored = OwnerDesktopReminders(path)
    assert restored.take_due(owner, now=due) == {'reminder': None}
    assert restored.list_open(owner)[0]['state'] == 'offered'
    with pytest.raises(ValueError): restored.acknowledge(owner, reminder['id'], '0' * 64)
    with pytest.raises(ValueError): restored.cancel(owner, reminder['id'])
    assert restored.acknowledge(owner, reminder['id'], reminder['acknowledgement_token']) == {'status': 'client_acknowledged'}
    assert restored.acknowledge(owner, reminder['id'], reminder['acknowledgement_token']) == {'status': 'client_acknowledged'}
    assert restored.list_open(owner) == []


def test_reminder_owner_and_cancellation_are_enforced(tmp_path):
    _, owner, reminders = _fixture(tmp_path)
    now = datetime.now(timezone.utc)
    due = (now + timedelta(minutes=1)).isoformat()
    with pytest.raises(PermissionError): reminders.create('other', 'private', due, 'a' * 32, now=now)
    record = reminders.create(owner, 'Fixture', due, 'b' * 32, now=now)
    with pytest.raises(PermissionError): reminders.take_due('other', now=now + timedelta(minutes=2))
    with pytest.raises(PermissionError): reminders.cancel('other', record['id'])
    assert reminders.cancel(owner, record['id']) == {'status': 'cancelled'}
    assert reminders.create(owner, 'Fixture', due, 'b' * 32, now=now + timedelta(minutes=2))['state'] == 'cancelled'
    assert reminders.take_due(owner, now=now + timedelta(minutes=2)) == {'reminder': None}
    with pytest.raises(ValueError): reminders.create(owner, 'past', now.isoformat(), 'c' * 32, now=now)
    recovery = reminders.create(owner, 'Recover fixture', due, 'e' * 32, now=now)
    assert reminders.take_due(owner, now=now + timedelta(minutes=2))['reminder']['id'] == recovery['id']
    with pytest.raises(PermissionError): reminders.resolve_unconfirmed('other', recovery['id'])
    assert reminders.list_open(owner)[0]['state'] == 'offered'
    assert reminders.resolve_unconfirmed(owner, recovery['id']) == {'status': 'owner_resolved'}
    assert reminders.take_due(owner, now=now + timedelta(minutes=2)) == {'reminder': None}
    assert reminders.list_open(owner) == []


def test_socket_authenticates_owner_and_rejects_forged_actor(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
    from alphonse.agent_v2.ipc import V2DaemonServer

    path, owner, reminders = _fixture(tmp_path)
    users = V2UserStore(path)
    # Older single-user installations may retain the owner role while still
    # identifying the same canonical administrator in user settings.
    users.update_user(owner, role='owner')
    users.create_user(display_name='Other owner-like role', user_id='other-owner', role='owner')
    with pytest.raises(PermissionError):
        reminders.create('other-owner', 'Private', (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(), '9' * 32)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    daemon = SimpleNamespace(experimental_owner_only=True, runtime=SimpleNamespace(user_store=users, queue=queue))
    socket_dir = tempfile.TemporaryDirectory(prefix='owner-reminder-', dir='/tmp')
    server = V2DaemonServer(daemon, os.path.join(socket_dir.name, 's'), experimental_owner_ingress=True)
    server.start()
    def request(method, params):
        with socket.socket(socket.AF_UNIX) as sock:
            sock.connect(str(server.socket_path))
            sock.sendall((json.dumps({'version': 1, 'request_id': 'fixture', 'method': method, 'params': params}) + '\n').encode())
            data = b''
            while b'\n' not in data:
                data += sock.recv(65536)
            return json.loads(data.split(b'\n')[0])
    try:
        due = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
        payload = {'message': 'Socket fixture', 'due_at': due, 'request_id': 'd' * 32}
        assert request('owner_reminder_create', {**payload, 'user': 'other'})['ok'] is False
        assert request('owner_reminder_create', payload)['ok'] is True
        assert request('owner_reminder_lookup', {'request_id': payload['request_id'], 'user': 'other'})['ok'] is False
        lookup = request('owner_reminder_lookup', {'request_id': payload['request_id']})
        assert lookup['ok'] is True and lookup['result']['reminder']['message'] == payload['message']
        listed = request('owner_reminder_list', {})
        assert listed['ok'] is True and len(listed['result']['reminders']) == 1
        assert request('owner_reminder_cancel', {'id': listed['result']['reminders'][0]['id']})['ok'] is True
        historical_now = datetime.now(timezone.utc) - timedelta(minutes=2)
        due = historical_now + timedelta(minutes=1)
        due_row = reminders.create(owner, 'Due fixture', due.isoformat(), 'f' * 32, now=historical_now)
        offered = request('owner_reminder_take', {})
        assert offered['ok'] and offered['result']['reminder']['id'] == due_row['id']
        # A lost response or restarted UI sees an uncertain offer and never
        # receives it again automatically; explicit owner review clears it.
        assert request('owner_reminder_take', {})['result']['reminder'] is None
        assert request('owner_reminder_list', {})['result']['reminders'][0]['state'] == 'offered'
        assert request('owner_reminder_resolve', {'id': due_row['id'], 'user': 'other'})['ok'] is False
        assert request('owner_reminder_resolve', {'id': due_row['id']})['result']['status'] == 'owner_resolved'
        assert request('owner_reminder_list', {})['result']['reminders'] == []
    finally:
        server.stop()
        socket_dir.cleanup()
