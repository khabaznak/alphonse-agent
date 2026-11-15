"""Owner schedule lookup through a disposable claim-bound worker."""
import json
import os
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_recovery import OwnerRecoveryUnavailable
from alphonse.agent_v2.core.intelligence.simple_loop.owner_schedules import execute_owner_schedules
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_search_memory_tool_definition, build_read_memory_source_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.owner_desktop_reminders import OwnerDesktopReminders
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def _worker(tmp_path, monkeypatch):
    class Provider:
        supports_image_input = False
        def __init__(self): self.requests = []
        def generate_json(self, request):
            self.requests.append(request)
            value = ({'kind': 'tool', 'tool_id': 'native.owner_schedules', 'arguments': {}, 'text': None}
                     if len(self.requests) == 1 else
                     {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'You have a garden reminder.'})
            return InferenceResult(json_value=value, raw_response=json.dumps(value),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    path = str(tmp_path / 'fixture.sqlite')
    users = V2UserStore(path)
    owner = users.onboard(display_name='Fixture owner', users_root=tmp_path / 'users')
    schedule_store = ScheduledTaskStore(path)
    future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    task = schedule_store.create_task(owner_user_id=owner.user_id, name='Water garden',
        prompt='Water the plants', schedule_kind='once', run_at=future)
    paused = schedule_store.create_task(owner_user_id=owner.user_id, name='Paused fixture',
        prompt='Owner paused reminder', schedule_kind='once', run_at=future)
    schedule_store.pause_task(paused.scheduled_task_id)
    schedule_store.create_task(owner_user_id='another-owner', name='Private other',
        prompt='Do not reveal', schedule_kind='once', run_at=future)
    OwnerDesktopReminders(path).create(owner.user_id, 'Desktop fixture', future, 'b' * 32)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    provider = Provider()
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(path),
        conversation_store=conversation, outbox=SQLiteOutboundStore(path), schedule_store=schedule_store,
        tools=registry, inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    daemon = V2Daemon(runtime, daemon_id='fixture', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    message_id = queue.enqueue_trusted_local_owner(prompt='What reminders do I have?', peer_uid=os.getuid())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = daemon.owner_message_status(message_id)
        if status not in {'pending', 'processing'}: break
        time.sleep(.01)
    return daemon, message_id, status, task, provider, conversation, owner


def test_owner_can_read_schedules_and_is_told_scheduler_is_paused(tmp_path, monkeypatch):
    daemon, message_id, status, _, provider, conversation, owner = _worker(tmp_path, monkeypatch)
    try:
        assert status == 'ready', status
        assert len(provider.requests) == 2
        assert 'Water garden' in provider.requests[1].prompt
        assert 'Paused fixture' in provider.requests[1].prompt
        assert 'Desktop fixture' in provider.requests[1].prompt
        assert 'Private other' not in provider.requests[1].prompt
        draft = daemon.take_owner_desktop_draft(message_id)
        assert 'Earlier scheduled work is paused' in draft['message']
        daemon.acknowledge_owner_desktop_draft(message_id, draft['acknowledgement_token'])
        assert conversation.list(owner_user_id=owner.user_id)[-1].content == draft['message']
        assert daemon.scheduler._thread is None
    finally:
        daemon.stop()


@pytest.mark.parametrize('restore', [False, True])
def test_changed_schedule_blocks_stale_draft_even_after_recovery(tmp_path, monkeypatch, restore):
    daemon, message_id, status, task, _, _, _ = _worker(tmp_path, monkeypatch)
    try:
        assert status == 'ready', status
        if restore:
            daemon._lock_file = object()
            daemon._owner_checkpoints.pop(message_id)
            daemon._owner_statuses.pop(message_id)
            assert daemon._restore_owner_ready(message_id)
        daemon.runtime.schedule_store.pause_task(task.scheduled_task_id)
        with pytest.raises((ValueError, RuntimeError, OwnerRecoveryUnavailable)):
            daemon.take_owner_desktop_draft(message_id)
    finally:
        daemon.stop()


def test_oversized_catalog_does_not_return_a_partial_list(tmp_path, monkeypatch):
    store = ScheduledTaskStore(str(tmp_path / 'schedules.sqlite'))
    future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    row = store.create_task(owner_user_id='owner', name='Fixture', prompt='Fixture', schedule_kind='once', run_at=future)
    monkeypatch.setattr(store, 'list_tasks', lambda **_: [row] * 101)
    context = SimpleNamespace(task=SimpleNamespace(user='owner'), schedule_store=store, cancellation_checker=lambda: False)
    assert execute_owner_schedules({}, context=context) == {'status': 'too_many', 'legacy_scheduler_running': False,
        'desktop_requires_open_conversation': True, 'schedules': [], 'desktop_reminders': []}
