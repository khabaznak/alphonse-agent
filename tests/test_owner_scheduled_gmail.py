"""Scheduled Gmail counts use the real worker with disposable accounts only."""
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from test_owner_gmail import f
from test_gmail_adapter import Wire
from test_gmail_unread import grant
from test_oauth_lifecycle import connect
from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.simple_loop.owner_gmail import OwnerGmailService
from alphonse.agent_v2.core.intelligence.simple_loop.owner_schedule_origin import create_schedule_origin_locked, stage_due_occurrence_locked
from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_ingress import enqueue_telegram_owner
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_read_memory_source_tool_definition, build_search_memory_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.owner_scheduled_delivery import OwnerScheduledDelivery
from alphonse.agent_v2.runtime import build_runtime_host


@pytest.mark.parametrize('channel', ['desktop', 'telegram'])
@pytest.mark.parametrize('revocation', ['none', 'before_read', 'before_publication'])
def test_scheduled_gmail_requires_live_occurrence_and_current_grant(f, tmp_path, monkeypatch, channel, revocation):
    connect(f); grant(f, 'one', 'grant')
    path = Path(f.accounts.users.db_path)
    users = f.accounts.users
    users.set_workspace_root(tmp_path / 'workspace')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    schedules = ScheduledTaskStore(path)
    original = 'Check my unread inbox count when this task is due.'
    if channel == 'telegram':
        users.bind_address(user_id='owner', integration_id='telegram-test', provider_key='telegram', provider_user_id='123')
        f.store.upsert(integration_id='telegram-test', provider_key='telegram', display_name='Telegram', enabled=True)
        source = enqueue_telegram_owner(queue, integration_id='telegram-test', update={'update_id': 73,
            'message': {'message_id': 73, 'text': original, 'from': {'id': 123},
                        'chat': {'id': 123, 'type': 'private'}}})
    else:
        source = queue.enqueue_trusted_local_owner(prompt=original, peer_uid=os.getuid())
    # Keep the original synthetic request claimed; the worker handles only the
    # separately fenced due occurrence. No external provider receives data.
    assert queue.claim_next() is not None
    now = datetime.now(timezone.utc)
    due = now + timedelta(minutes=1)
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        record = schedules.create_task(owner_user_id='owner', name='Synthetic inbox check', prompt=original,
            schedule_kind='once', run_at=due.isoformat(), delivery_mode='pdca',
            idempotency_key='owner-schedule:' + source, now=now, connection=conn)
        create_schedule_origin_locked(conn, SimpleNamespace(message_id=source, user='owner', goal=original), record)
        occurrence = stage_due_occurrence_locked(conn, record.scheduled_task_id, now=due)
    queue._owner_schedule_clock = lambda: due
    wire = Wire({'id': 'INBOX', 'messagesUnread': 4})
    if revocation == 'before_read':
        f.accounts.revoke_grant(f.owner, grant_id='grant', expected_revision=1)

    if revocation == 'before_publication':
        from alphonse.agent_v2.core.intelligence.simple_loop import owner_read_recovery, owner_telegram_scheduled_delivery
        original_restore = owner_read_recovery.restore_owner_read_observations
        revoked = []
        def revoke_after_restore(*args, **kwargs):
            result = original_restore(*args, **kwargs)
            if kwargs.get('allow_completed') and not revoked:
                f.accounts.revoke_grant(f.owner, grant_id='grant', expected_revision=1)
                revoked.append(True)
            return result
        monkeypatch.setattr(owner_read_recovery, 'restore_owner_read_observations', revoke_after_restore)
        monkeypatch.setattr(owner_telegram_scheduled_delivery, 'restore_owner_read_observations', revoke_after_restore)

    class Provider:
        supports_image_input = False
        calls = 0
        def generate_json(self, request):
            self.calls += 1
            assert 'native.owner_gmail_unread' in request.prompt
            result = ({'kind': 'tool', 'tool_id': 'native.owner_gmail_unread', 'arguments': {}, 'text': None}
                      if self.calls == 1 else {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Done.'})
            return InferenceResult(json_value=result, raw_response=str(result), usage={'input_tokens': 100, 'output_tokens': 20})

    class Listener:
        def __init__(self, *_): self.sent = []
        def start(self): pass
        def stop(self, **_): return True
        def send_reply(self, **values): self.sent.append(values); return 'provider-id'
    from alphonse.agent_v2.core.intelligence.simple_loop import owner_telegram_listener
    listener = Listener()
    monkeypatch.setattr(owner_telegram_listener, 'TelegramOwnerListener', lambda *_: listener)
    tools = InMemoryToolRegistry()
    tools.register(build_search_memory_tool_definition()); tools.register(build_read_memory_source_tool_definition())
    runtime = build_runtime_host(user_store=users, integration_store=f.store, messages=queue,
        question_store=SQLiteQuestionStore(path), schedule_store=schedules,
        conversation_store=conversation, outbox=SQLiteOutboundStore(path), tools=tools,
        inference=InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'fixture', 'offline')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    runtime.owner_gmail_service = OwnerGmailService(f.accounts, lambda _: f.service, transport=wire)
    daemon = V2Daemon(runtime, daemon_id='scheduled-gmail-test', experimental_owner_only=True,
                      owner_telegram_enabled=channel == 'telegram')
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None); monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            with queue._connect() as conn:
                state = conn.execute('SELECT state FROM v2_owner_schedule_occurrences WHERE message_id=?', (occurrence,)).fetchone()['state']
            if state in {'completed', 'failed'} or daemon._owner_statuses.get(occurrence) == 'failed': break
            time.sleep(.02)
        if revocation == 'none':
            assert state == 'completed'
            assert len(wire.calls) == 1
            pending = OwnerScheduledDelivery(path).list_pending('owner')
            assert len(pending) == 1 and 'Gmail INBOX unread messages: 4' in pending[0]['message']
            if channel == 'telegram':
                assert len(listener.sent) == 1 and 'Gmail INBOX unread messages: 4' in listener.sent[0]['text']
        else:
            assert daemon._owner_statuses.get(occurrence) == 'failed'
            assert state in {'queued', 'failed'}  # blocked/uncertain, never completed
            assert len(wire.calls) == (0 if revocation == 'before_read' else 1)
            assert not OwnerScheduledDelivery(path).list_pending('owner')
            assert not listener.sent
    finally:
        daemon.stop()
