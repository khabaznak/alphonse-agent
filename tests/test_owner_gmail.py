"""Direct-owner Gmail count, with disposable accounts and fake transport only."""
from pathlib import Path
import json
import os
import time

import pytest

from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.simple_loop.owner_gmail import OwnerGmailService
from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_recovery import OwnerRecoveryUnavailable
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_search_memory_tool_definition, build_read_memory_source_tool_definition
from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from test_gmail_adapter import Wire
from test_gmail_unread import grant
from test_oauth_lifecycle import MemoryBackend, FakeProvider, connect, expire


@pytest.fixture
def f(tmp_path):
    """Use the production topology: user, queue and account grants in one DB."""
    from types import SimpleNamespace
    from alphonse.agent_v2.accounts import AccountService, AccountSession
    from alphonse.agent_v2.integrations.credentials import CredentialVault
    from alphonse.agent_v2.integrations.oauth import ClientConfig, OAuthService, SCOPES
    from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
    from alphonse.agent_v2.users import V2UserStore
    path = tmp_path / 'fixture.sqlite'
    users = V2UserStore(path)
    users.onboard(display_name='Owner', user_id='owner', users_root=tmp_path / 'profiles')
    users.create_user(display_name='Other', user_id='other')
    backend = MemoryBackend()
    store = SQLiteIntegrationStore(path, credential_vault=CredentialVault(backend))
    accounts = AccountService(store, users)
    configs = [ClientConfig(p, 'public-app', 'http://127.0.0.1:54321/callback') for p in SCOPES]
    service = OAuthService(accounts, configs, client_factory=FakeProvider)
    return SimpleNamespace(service=service, accounts=accounts, store=store, backend=backend,
                           owner=AccountSession('owner'), other=AccountSession('other'), configs=configs)


def start_worker(f, tmp_path, monkeypatch, wire, *, ask_question=False, unit=None):
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    path = Path(f.accounts.users.db_path)
    monkeypatch.setenv('ALPHONSE_V2_DB_PATH', str(path))
    class Provider:
        supports_image_input = False
        def __init__(self): self.requests = []
        def generate_json(self, request):
            self.requests.append(request)
            action = ({'kind': 'tool', 'tool_id': 'native.owner_gmail_unread',
                       'arguments': {} if unit is None else {'unit': unit}, 'text': None}
                      if len(self.requests) == 1 else
                      {'kind': 'question' if ask_question else 'answer', 'tool_id': None, 'arguments': None,
                       'text': 'Which mailbox?' if ask_question else 'There are 99 unread messages.'})
            return InferenceResult(json_value=action, raw_response=json.dumps(action),
                                   usage={'input_tokens': 100, 'output_tokens': 20})
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    tools = InMemoryToolRegistry()
    tools.register(build_search_memory_tool_definition())
    tools.register(build_read_memory_source_tool_definition())
    provider = Provider()
    runtime = build_runtime_host(user_store=f.accounts.users, integration_store=f.store, messages=queue,
        question_store=SQLiteQuestionStore(path), conversation_store=conversation,
        outbox=SQLiteOutboundStore(path), tools=tools,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    runtime.owner_gmail_service = OwnerGmailService(f.accounts, lambda _: f.service, transport=wire)
    daemon = V2Daemon(runtime, daemon_id='fixture', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    message_id = queue.enqueue_trusted_local_owner(prompt='How many unread inbox messages?', peer_uid=os.getuid())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = daemon.owner_message_status(message_id)
        if status not in {'pending', 'processing'}: break
        time.sleep(.01)
    return daemon, message_id, status, provider


@pytest.mark.parametrize('revoke_before_answer', [False, True])
def test_telegram_owner_can_read_authorized_gmail_and_send_verified_count(f, tmp_path, monkeypatch, revoke_before_answer):
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_ingress import enqueue_telegram_owner
    from alphonse.agent_v2.core.intelligence.simple_loop import owner_telegram_listener as module
    connect(f); grant(f, 'one', 'grant')
    f.accounts.users.bind_address(user_id='owner', integration_id='telegram-test', provider_key='telegram', provider_user_id='123')
    f.store.upsert(integration_id='telegram-test', provider_key='telegram', display_name='Telegram', enabled=True)
    path = Path(f.accounts.users.db_path)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    tools = InMemoryToolRegistry()
    tools.register(build_search_memory_tool_definition())
    tools.register(build_read_memory_source_tool_definition())
    wire = Wire({'id': 'INBOX', 'messagesUnread': 4})
    sent = []
    class Listener:
        def __init__(self, *_): pass
        def start(self): pass
        def stop(self, **_): return True
        def send_reply(self, **values): sent.append(values); return '99'
    monkeypatch.setattr(module, 'TelegramOwnerListener', Listener)
    class Provider:
        supports_image_input = False
        calls = 0
        def generate_json(self, request):
            self.calls += 1
            assert 'native.owner_gmail_unread' in request.prompt
            if self.calls == 2 and revoke_before_answer:
                f.accounts.revoke_grant(f.owner, grant_id='grant', expected_revision=1)
            value = ({'kind': 'tool', 'tool_id': 'native.owner_gmail_unread', 'arguments': {}, 'text': None}
                     if self.calls == 1 else {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Done.'})
            return InferenceResult(json_value=value, raw_response=json.dumps(value),
                                   usage={'input_tokens': 100, 'output_tokens': 20})
    provider = Provider()
    runtime = build_runtime_host(user_store=f.accounts.users, integration_store=f.store, messages=queue,
        question_store=SQLiteQuestionStore(path), conversation_store=conversation, outbox=SQLiteOutboundStore(path),
        tools=tools, inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    runtime.owner_gmail_service = OwnerGmailService(f.accounts, lambda _: f.service, transport=wire)
    daemon = V2Daemon(runtime, daemon_id='telegram-gmail-fixture', experimental_owner_only=True, owner_telegram_enabled=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        mid = enqueue_telegram_owner(queue, integration_id='telegram-test', update={'update_id': 123,
            'message': {'message_id': 123, 'text': 'How many unread inbox messages?', 'from': {'id': 123},
                        'chat': {'id': 123, 'type': 'private'}}})
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and not sent and daemon._owner_statuses.get(mid) != 'failed':
            time.sleep(.02)
        if revoke_before_answer:
            assert daemon._owner_statuses.get(mid) == 'failed'
            assert not sent
            assert len(wire.calls) == 1
            return
        assert sent and sent[0]['chat_id'] == '123'
        assert 'Gmail INBOX unread messages: 4' in sent[0]['text']
        assert len(wire.calls) == 1
        with queue._connect() as conn:
            assert conn.execute('SELECT status FROM v2_owner_telegram_replies WHERE message_id=?', (mid,)).fetchone()[0] in ('provider_accepted', 'finalized')
    finally:
        daemon.stop()


@pytest.mark.parametrize('unit, count, expected', [
    ('threads', 1, 'thread'), ('threads', 3, 'threads'), ('messages', 1, 'message'),
])
def test_owner_unread_counts_keep_unit_and_do_not_refetch(f, tmp_path, monkeypatch, unit, count, expected):
    connect(f); grant(f, 'one', 'grant')
    wire = Wire({'id': 'INBOX', 'messagesUnread': count if unit == 'messages' else 99,
                 'threadsUnread': count if unit == 'threads' else 99})
    daemon, message_id, status, _ = start_worker(f, tmp_path, monkeypatch, wire, unit=unit)
    try:
        assert status == 'ready'
        daemon._lock_file = object()
        daemon._owner_checkpoints.pop(message_id)
        daemon._owner_statuses.pop(message_id)
        assert daemon._restore_owner_ready(message_id)
        draft = daemon.take_owner_desktop_draft(message_id)
        assert f'Gmail INBOX unread {expected}: {count}' in draft['message']
        assert '99 unread' not in draft['message'] and len(wire.calls) == 1
    finally:
        daemon.stop()


@pytest.mark.parametrize('arguments', [{'unit': 'accounts'}, {'unit': None}, {'unit': ['threads']},
                                       {'unit': 'threads', 'connection_id': 'other'}])
def test_owner_count_rejects_invalid_units_or_model_supplied_authority(arguments):
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_gmail import owner_gmail_unit
    with pytest.raises(ValueError, match='owner_gmail_arguments_invalid'):
        owner_gmail_unit(arguments)


@pytest.mark.parametrize('restore', [False, True])
def test_owner_count_uses_authorized_snapshot_and_does_not_refetch_on_pickup(f, tmp_path, monkeypatch, restore):
    f.accounts.users.update_user('owner', role='owner')
    connect(f); grant(f, 'one', 'grant')
    wire = Wire({'id': 'INBOX', 'messagesUnread': 4})
    daemon, message_id, status, provider = start_worker(f, tmp_path, monkeypatch, wire)
    try:
        assert status == 'ready', status
        assert len(wire.calls) == 1 and wire.calls[0][0] == 'labels/INBOX'
        assert 'disposable-access' not in provider.requests[1].prompt
        assert 'grant_id' not in provider.requests[1].prompt
        if restore:
            daemon._lock_file = object()
            daemon._owner_checkpoints.pop(message_id)
            daemon._owner_statuses.pop(message_id)
            assert daemon._restore_owner_ready(message_id)
        draft = daemon.take_owner_desktop_draft(message_id)
        assert 'Gmail INBOX unread messages: 4' in draft['message']
        assert 'There are 99 unread messages.' not in draft['message']
        assert len(wire.calls) == 1
    finally:
        daemon.stop()


def test_grant_revocation_before_pickup_hides_count_without_refetch(f, tmp_path, monkeypatch):
    connect(f); grant(f, 'one', 'grant')
    wire = Wire({'id': 'INBOX', 'messagesUnread': 4})
    daemon, message_id, status, _ = start_worker(f, tmp_path, monkeypatch, wire)
    try:
        assert status == 'ready'
        f.accounts.revoke_grant(f.owner, grant_id='grant', expected_revision=1)
        with pytest.raises((OwnerRecoveryUnavailable, ValueError, RuntimeError)):
            daemon.take_owner_desktop_draft(message_id)
        assert len(wire.calls) == 1
    finally:
        daemon.stop()


def test_restart_after_grant_revocation_cannot_restore_count(f, tmp_path, monkeypatch):
    connect(f); grant(f, 'one', 'grant')
    wire = Wire({'id': 'INBOX', 'messagesUnread': 4})
    daemon, message_id, status, _ = start_worker(f, tmp_path, monkeypatch, wire)
    try:
        assert status == 'ready'
        daemon._lock_file = object()
        daemon._owner_checkpoints.pop(message_id)
        daemon._owner_statuses.pop(message_id)
        f.accounts.revoke_grant(f.owner, grant_id='grant', expected_revision=1)
        # Restoring the inert ready marker may succeed; pickup revalidates
        # current authority before any count can leave the daemon.
        assert daemon._restore_owner_ready(message_id) is True
        with pytest.raises((OwnerRecoveryUnavailable, ValueError, RuntimeError)):
            daemon.take_owner_desktop_draft(message_id)
        assert len(wire.calls) == 1
    finally:
        daemon.stop()


def test_question_after_gmail_observation_is_not_delivered_as_count(f, tmp_path, monkeypatch):
    connect(f); grant(f, 'one', 'grant')
    wire = Wire({'id': 'INBOX', 'messagesUnread': 4})
    daemon, message_id, status, _ = start_worker(f, tmp_path, monkeypatch, wire, ask_question=True)
    try:
        assert status != 'ready'
        with pytest.raises((OwnerRecoveryUnavailable, ValueError, RuntimeError)):
            daemon.take_owner_desktop_draft(message_id)
        assert len(wire.calls) == 1
    finally:
        daemon.stop()


def test_expired_token_does_not_call_provider(f, tmp_path, monkeypatch):
    connect(f); grant(f, 'one', 'grant'); expire(f)
    wire = Wire({'id': 'INBOX', 'messagesUnread': 4})
    daemon, message_id, status, _ = start_worker(f, tmp_path, monkeypatch, wire)
    try:
        assert status != 'ready'
        assert not wire.calls and f.service.clients['google'].refresh_count == 0
    finally:
        daemon.stop()


def test_ambiguous_accounts_do_not_call_provider(f, tmp_path, monkeypatch):
    connect(f, connection='one'); grant(f, 'one', 'grant-one')
    connect(f, connection='two'); grant(f, 'two', 'grant-two')
    wire = Wire({'id': 'INBOX', 'messagesUnread': 4})
    daemon, message_id, status, _ = start_worker(f, tmp_path, monkeypatch, wire)
    try:
        assert status != 'ready'
        assert not wire.calls
        with pytest.raises((OwnerRecoveryUnavailable, ValueError, RuntimeError)):
            daemon.take_owner_desktop_draft(message_id)
    finally:
        daemon.stop()


@pytest.mark.parametrize('change', ['revoked', 'another_account', 'credentials_changed', 'observation_expired'])
def test_pickup_transaction_rechecks_changes_after_preflight(f, tmp_path, monkeypatch, change):
    from datetime import datetime, timedelta
    from alphonse.agent_v2.core.intelligence.simple_loop import owner_gmail, owner_read_recovery
    connect(f); grant(f, 'one', 'grant')
    wire = Wire({'id': 'INBOX', 'messagesUnread': 4})
    daemon, message_id, status, _ = start_worker(f, tmp_path, monkeypatch, wire)
    try:
        assert status == 'ready'
        original = owner_read_recovery.restore_owner_read_observations
        def interleave(*args, **kwargs):
            value = original(*args, **kwargs)
            if change == 'revoked':
                f.accounts.revoke_grant(f.owner, grant_id='grant', expected_revision=1)
            elif change == 'another_account':
                connect(f, connection='two'); grant(f, 'two', 'grant-two')
            elif change == 'credentials_changed':
                record = f.store.get('one')
                token = f.store.resolve_credentials(record)
                f.store.install_credentials('one', token, expected_generation=record.credential_generation)
            else:
                class Later(datetime):
                    @classmethod
                    def now(cls, tz=None):
                        return datetime.now(tz) + timedelta(minutes=6)
                monkeypatch.setattr(owner_gmail, 'datetime', Later)
            return value
        monkeypatch.setattr(owner_read_recovery, 'restore_owner_read_observations', interleave)
        with pytest.raises((OwnerRecoveryUnavailable, ValueError, RuntimeError)):
            daemon.take_owner_desktop_draft(message_id)
        assert len(wire.calls) == 1
    finally:
        daemon.stop()
