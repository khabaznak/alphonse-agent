"""Clock evidence through the real opt-in worker, using disposable stores."""
import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_recovery import OwnerRecoveryUnavailable
from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_search_memory_tool_definition, build_read_memory_source_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def run_worker(tmp_path, monkeypatch, *, use_clock=True, zone='America/Mexico_City', invalid_zone=False, owner_role='admin'):
    class Provider:
        supports_image_input = False
        calls = 0
        def generate_json(self, request):
            self.calls += 1
            if use_clock and self.calls == 1:
                value = {'kind': 'tool', 'tool_id': 'native.owner_clock', 'arguments': {}, 'text': None}
            else:
                value = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic response.'}
            return InferenceResult(json_value=value, raw_response=json.dumps(value),
                                   usage={'input_tokens': 100, 'output_tokens': 20})
    path = tmp_path / 'fixture.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Fixture owner', users_root=tmp_path / 'users')
    if owner_role != 'admin':
        owner = users.update_user(owner.user_id, role=owner_role)
    users.set_timezone(zone, owner.user_id)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    tools = InMemoryToolRegistry()
    tools.register(build_search_memory_tool_definition())
    tools.register(build_read_memory_source_tool_definition())
    router = InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'fixture', 'offline'))
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(path),
        conversation_store=conversation, outbox=SQLiteOutboundStore(path), tools=tools, inference=router,
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    daemon = V2Daemon(runtime, daemon_id='fixture', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    if invalid_zone:
        monkeypatch.setattr(users, 'timezone', lambda _: 'Invalid/Fixture')
    daemon.start()
    message_id = queue.enqueue_trusted_local_owner(prompt='Fixture prompt', peer_uid=__import__('os').getuid())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = daemon.owner_message_status(message_id)
        if status not in {'pending', 'processing'}:
            break
        time.sleep(.01)
    return daemon, message_id, status, users, owner, conversation


def test_worker_delivers_clock_observation_and_records_exact_message(tmp_path, monkeypatch):
    daemon, message_id, status, users, owner, conversation = run_worker(tmp_path, monkeypatch)
    try:
        assert status == 'ready', status
        draft = daemon.take_owner_desktop_draft(message_id)
        assert draft['message'].startswith('Synthetic response.\n\nHost clock observed as of: ')
        assert 'America/Mexico_City' in draft['message']
        daemon.acknowledge_owner_desktop_draft(message_id, draft['acknowledgement_token'])
        daemon.acknowledge_owner_desktop_draft(message_id, draft['acknowledgement_token'])
        assert conversation.list(owner_user_id=owner.user_id)[-1].content == draft['message']
    finally:
        daemon.stop()


@pytest.mark.parametrize('change', ['zone', 'expired', 'cancel'])
def test_pickup_revalidates_clock_evidence(tmp_path, monkeypatch, change):
    daemon, message_id, status, users, owner, _ = run_worker(tmp_path, monkeypatch)
    try:
        assert status == 'ready', status
        if change == 'zone':
            users.set_timezone('UTC', owner.user_id)
        elif change == 'expired':
            from alphonse.agent_v2.core.intelligence.simple_loop import owner_clock
            class Later(datetime):
                @classmethod
                def now(cls, tz=None):
                    return datetime.now(tz) + timedelta(minutes=6)
            monkeypatch.setattr(owner_clock, 'datetime', Later)
        else:
            daemon._owner_checkpoints[message_id][1].cancellation_checker = lambda: True
        with pytest.raises((ValueError, RuntimeError, OwnerRecoveryUnavailable)):
            daemon.take_owner_desktop_draft(message_id)
    finally:
        daemon.stop()


def test_unrelated_answer_does_not_need_clock_or_timezone(tmp_path, monkeypatch):
    daemon, message_id, status, users, owner, conversation = run_worker(tmp_path, monkeypatch, use_clock=False)
    try:
        assert status == 'ready', status
        users.set_timezone('UTC', owner.user_id)
        draft = daemon.take_owner_desktop_draft(message_id)
        assert draft['message'] == 'Synthetic response.'
    finally:
        daemon.stop()


def test_invalid_timezone_does_not_produce_clock_answer(tmp_path, monkeypatch):
    daemon, message_id, status, *_ = run_worker(tmp_path, monkeypatch, invalid_zone=True)
    try:
        assert status != 'ready'
        with pytest.raises((ValueError, RuntimeError, OwnerRecoveryUnavailable)):
            daemon.take_owner_desktop_draft(message_id)
    finally:
        daemon.stop()


def test_clock_evidence_survives_ready_handoff_restore(tmp_path, monkeypatch):
    daemon, message_id, status, _, _, _ = run_worker(tmp_path, monkeypatch)
    try:
        assert status == 'ready'
        # Reconstruct from the persisted ready handoff as a replacement worker does.
        daemon._lock_file = object()  # Fixture stands in for the daemon's exclusive lock.
        daemon._owner_checkpoints.pop(message_id)
        daemon._owner_statuses.pop(message_id)
        assert daemon._restore_owner_ready(message_id)
        _, restored_context = daemon._owner_checkpoints[message_id]
        assert restored_context.skill_store is daemon.runtime.core.skill_store
        assert restored_context.artifact_resources is daemon.runtime.core.artifact_resources
        assert restored_context.skill_bindings is daemon.runtime.core.skill_bindings
        draft = daemon.take_owner_desktop_draft(message_id)
        assert 'Host clock observed as of:' in draft['message']
        assert 'America/Mexico_City' in draft['message']
    finally:
        daemon.stop()


def test_canonical_active_owner_role_can_complete_and_acknowledge(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    monkeypatch.setenv('ALPHONSE_V2_DB_PATH', str(tmp_path / 'owner.sqlite'))
    daemon, message_id, status, _, owner, conversation = run_worker(tmp_path, monkeypatch, owner_role='owner')
    try:
        assert status == 'ready'
        assert owner.role == 'owner'
        draft = daemon.take_owner_desktop_draft(message_id)
        daemon.acknowledge_owner_desktop_draft(message_id, draft['acknowledgement_token'])
        daemon.acknowledge_owner_desktop_draft(message_id, draft['acknowledgement_token'])
        assert conversation.list(owner_user_id=owner.user_id)[-1].content == draft['message']
    finally:
        daemon.stop()
