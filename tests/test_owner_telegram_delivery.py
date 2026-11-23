"""The new Telegram ledger never drains old messages or repeats uncertain sends."""
import sqlite3
import threading

import pytest

from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_delivery import (
    TelegramReplyUnavailable, dispatch_once, ensure_schema, stage_locked,
)


@pytest.fixture
def reply():
    return {'message_id': 'task-1', 'owner_id': 'owner', 'integration_id': 'telegram-home',
            'provider_user_id': '42', 'chat_id': '42', 'provider_message_id': '11', 'text': 'Hello'}


def stage(path, reply):
    with sqlite3.connect(path) as conn:
        ensure_schema(conn)
        conn.execute('BEGIN IMMEDIATE')
        stage_locked(conn, reply, verify=lambda _conn, _reply: True)


def test_delivers_once_and_never_drains_other_pending_messages(tmp_path, reply):
    path = str(tmp_path / 'queue.sqlite')
    stage(path, reply)
    other = {**reply, 'message_id': 'old-message'}
    stage(path, other)
    calls = []
    send = lambda **kw: calls.append(kw) or '99'
    assert dispatch_once(path, 'task-1', verify=lambda *_: True, send=send) == 'provider_accepted'
    assert dispatch_once(path, 'task-1', verify=lambda *_: True, send=send) == 'provider_accepted'
    assert calls == [{'integration_id': 'telegram-home', 'chat_id': '42', 'text': 'Hello'}]
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT status FROM v2_owner_telegram_replies WHERE message_id='old-message'").fetchone()[0] == 'pending'


@pytest.mark.parametrize('outcome', ['exception', 'missing_ack'])
def test_uncertain_send_is_never_replayed(tmp_path, reply, outcome):
    path = str(tmp_path / 'queue.sqlite')
    stage(path, reply)
    sent = []
    def send(**kwargs):
        sent.append(kwargs)
        if outcome == 'exception':
            raise TimeoutError('provider result unknown')
        return ''
    assert dispatch_once(path, 'task-1', verify=lambda *_: True, send=send) == 'started_unconfirmed'
    assert dispatch_once(path, 'task-1', verify=lambda *_: True, send=send) == 'started_unconfirmed'
    assert len(sent) == 1


def test_conflicting_draft_and_revoked_destination_do_not_send(tmp_path, reply):
    path = str(tmp_path / 'queue.sqlite')
    stage(path, reply)
    with sqlite3.connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        with pytest.raises(TelegramReplyUnavailable):
            stage_locked(conn, {**reply, 'chat_id': '88'}, verify=lambda *_: True)
    calls = []
    with pytest.raises(TelegramReplyUnavailable):
        dispatch_once(path, 'task-1', verify=lambda *_: False, send=lambda **kw: calls.append(kw))
    assert calls == []


def test_staging_requires_an_authenticated_draft_in_a_transaction(tmp_path, reply):
    with sqlite3.connect(tmp_path / 'queue.sqlite') as conn:
        ensure_schema(conn)
        with pytest.raises(TelegramReplyUnavailable):
            stage_locked(conn, reply, verify=lambda *_: True)
        conn.execute('BEGIN IMMEDIATE')
        with pytest.raises(TelegramReplyUnavailable):
            stage_locked(conn, reply, verify=lambda *_: False)


def test_concurrent_dispatch_never_sends_twice(tmp_path, reply):
    path = str(tmp_path / 'queue.sqlite')
    stage(path, reply)
    started, release = threading.Event(), threading.Event()
    calls = []
    def send(**kwargs):
        calls.append(kwargs)
        started.set()
        assert release.wait(3)
        return '99'
    result = []
    worker = threading.Thread(target=lambda: result.append(dispatch_once(path, 'task-1', verify=lambda *_: True, send=send)))
    worker.start()
    try:
        assert started.wait(3)
        assert dispatch_once(path, 'task-1', verify=lambda *_: True, send=send) == 'started_unconfirmed'
    finally:
        release.set()
        worker.join(timeout=3)
    assert result == ['provider_accepted']
    assert len(calls) == 1
