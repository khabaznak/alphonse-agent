"""Real opt-in local daemon path with disposable storage and inference."""
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace
import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_read_memory_source_tool_definition, build_search_memory_tool_definition
from alphonse.agent_v2.daemon import V2Daemon, _owner_public_failure_code
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def test_owner_failure_codes_are_allowlisted_and_do_not_disclose_errors():
    assert _owner_public_failure_code('candidate_observation_incomplete') == 'source_incomplete'
    assert _owner_public_failure_code('candidate_deadline_exceeded') == 'time_limit'
    assert _owner_public_failure_code('raw private source text') == 'needs_attention'
    assert _owner_public_failure_code(None) == 'needs_attention'
    assert _owner_public_failure_code('candidate_deadline_exceeded', cancelled=True) == 'cancel_requested'


@pytest.mark.parametrize('old_flag', [None, '0', '1'])
def test_daemon_main_always_selects_simple_loop(monkeypatch, tmp_path, old_flag):
    from alphonse.agent_v2 import daemon as module
    if old_flag is None:
        monkeypatch.delenv('ALPHONSE_OWNER_ONLY_SIMPLE_LOOP', raising=False)
    else:
        monkeypatch.setenv('ALPHONSE_OWNER_ONLY_SIMPLE_LOOP', old_flag)
    monkeypatch.setattr(module, '_claim_single_instance_lock', lambda _: object())
    monkeypatch.setattr(module, 'migrate_legacy_databases', lambda: {})
    monkeypatch.setattr(module, 'default_database_path', lambda: tmp_path / 'fixture.sqlite')
    monkeypatch.setattr(module.signal, 'signal', lambda *_: None)
    user = SimpleNamespace(users_root=tmp_path, admin_user=lambda: SimpleNamespace(user_id='owner'),
                           agent_config_root=lambda: tmp_path)
    monkeypatch.setattr(module, 'V2UserStore', SimpleNamespace(default=lambda: user))
    asset = SimpleNamespace(migrate_to_user_directories=lambda: None)
    monkeypatch.setattr(module, 'SQLiteAssetStore', SimpleNamespace(default=lambda **_: asset))
    for name in ('SQLiteQuestionStore', 'ProjectStore', 'ScheduledTaskStore',
                 'SQLiteWebToolsSettingsStore', 'SQLiteCodeModeSettingsStore', 'SQLiteMediaToolsSettingsStore',
                 'SQLiteArtifactStore', 'SQLiteConversationStore', 'SQLiteMemorySettingsStore',
                 'SQLiteSystemOneSettingsStore', 'SQLiteIntelligenceEngineSettingsStore', 'SQLiteOutboundStore',
                 'SQLiteIntegrationStore', 'SQLiteInferenceSettingsStore', 'SQLiteProjectSessionStore',
                 'SQLiteMemorySessionStore', 'SQLiteCommunicationThreadStore'):
        monkeypatch.setattr(module, name, SimpleNamespace(default=lambda: object()))
    monkeypatch.setattr(module, 'AgentConfigStore', lambda *_: object())
    class Queue:
        @classmethod
        def default(cls, *, lease_owner):
            return ('legacy', lease_owner)
        def __new__(cls, path, *, lease_owner, experimental_claim_fencing):
            return ('owner', path, lease_owner, experimental_claim_fencing)
    monkeypatch.setattr(module, 'SQLiteMessageQueue', Queue)
    captured = {}
    def runtime(**kwargs):
        captured['runtime'] = kwargs
        return object()
    monkeypatch.setattr(module, 'build_runtime_host', runtime)
    class Daemon:
        def __init__(self, runtime, **kwargs):
            captured['daemon'] = kwargs
        def run_forever(self):
            captured['ran'] = True
    monkeypatch.setattr(module, 'V2Daemon', Daemon)
    module.main()
    assert captured['ran']
    assert captured['daemon']['experimental_owner_only'] is True
    assert captured['runtime']['experimental_owner_read'] is True
    assert captured['runtime']['messages'][0] == 'owner'
    assert callable(captured['runtime']['owner_selected_draft_sink'])


@pytest.mark.parametrize('mode', ['answer', 'question', 'question_context_changed', 'cancel', 'abandon', 'unknown_effect', 'unknown_outbox', 'bad_schema', 'bad_status'])
def test_explicit_owner_worker_routes_to_socket_and_saves_history(tmp_path, monkeypatch, mode):
    holder = {}
    class Provider:
        supports_image_input = False
        def __init__(self, source_id):
            self.source_id = source_id
            self.calls = 0
        def generate_json(self, request):
            self.calls += 1
            self.last_prompt = request.prompt
            if self.calls == 1:
                value = {'kind': 'tool', 'tool_id': 'native.search_memory', 'arguments': {
                    'query': 'apples', 'source_types': ['conversation']}, 'text': None}
            elif self.calls == 2:
                value = {'kind': 'tool', 'tool_id': 'native.read_memory_source', 'arguments': {
                    'source_id': self.source_id, 'neighbors': 0}, 'text': None}
            else:
                if self.calls == 3:
                    assert 'Synthetic apples.' in request.prompt
                if mode == 'cancel':
                    outcome = holder['daemon'].trigger_killswitch(actor_user_id=holder['owner_id'])
                    assert outcome['status'] == 'cancel_requested'
                    assert outcome['notification_error'] == 'owner_notice_unavailable'
                question = mode in {'question', 'question_context_changed'} and self.calls == 3
                value = {'kind': 'question' if question else 'answer', 'tool_id': None,
                         'arguments': None, 'text': 'Which apples?' if question else 'Synthetic answer.'}
            return InferenceResult(json_value=value, raw_response=json.dumps(value),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    holder['owner_id'] = owner.user_id
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    source = conversation.record(owner_user_id=owner.user_id, role='user', content='Synthetic apples.',
                                 source='test', source_message_id='source-fixture')
    tools = InMemoryToolRegistry()
    tools.register(build_search_memory_tool_definition())
    tools.register(build_read_memory_source_tool_definition())
    provider = Provider('conversation:' + source.event_id)
    router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline'))
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(path),
        conversation_store=conversation, outbox=SQLiteOutboundStore(path), tools=tools, inference=router, coding_skill_service=object(),
        experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    daemon = V2Daemon(runtime, daemon_id='fixture', experimental_owner_only=True)
    holder['daemon'] = daemon
    # The exclusive owner worker must retain the same protected package and
    # account-binding services as the runtime. This alone exposes no skills:
    # the candidate still restricts which tools and observations it accepts.
    candidate = runtime.processor.owner_read_candidate
    original_process = candidate.process
    def process_with_runtime_context(task, context):
        assert context.skill_store is runtime.core.skill_store
        assert context.artifact_resources is runtime.core.artifact_resources
        assert context.skill_bindings is runtime.core.skill_bindings
        execution = context.tool_execution_context(task)
        assert execution.skill_store is context.skill_store
        assert execution.artifact_resources is context.artifact_resources
        assert execution.skill_bindings is context.skill_bindings
        return original_process(task, context)
    monkeypatch.setattr(candidate, 'process', process_with_runtime_context)
    register_checkpoint = daemon.register_owner_checkpoint
    def register_and_check_not_yet_pickable(checkpoint, context):
        register_checkpoint(checkpoint, context)
        # The draft is registered before processor/cancellation handoff ends;
        # a same-UID client must not be able to take it during that window.
        with pytest.raises(ValueError, match='owner_desktop_pickup_unavailable'):
            daemon.take_owner_desktop_draft(checkpoint.task.message_id)
    monkeypatch.setattr(daemon, 'register_owner_checkpoint', register_and_check_not_yet_pickable)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    with tempfile.TemporaryDirectory(prefix='owner-e2e-', dir='/private/tmp') as socket_dir:
        daemon.ipc = V2DaemonServer(daemon, os.path.join(socket_dir, 'p.sock'))
        daemon.start()
        try:
            # The replacement owner-only runtime does not execute scheduled
            # work yet; starting it must not start the generic scheduler.
            assert daemon.scheduler._thread is None
            client = V2DaemonClient(daemon.ipc.socket_path)
            assert client.request('experimental_owner_status')['processing_ready'] is True
            with monkeypatch.context() as patch:
                patch.setattr(daemon, 'owner_processing_ready', lambda: False)
                with pytest.raises(RuntimeError, match='owner_worker_unavailable'):
                    client.request('experimental_owner_ingress', prompt='Would be stranded')
                assert queue.size() == 0
            with pytest.raises(RuntimeError, match='owner_only_use_owner_conversation'):
                client.request('queue_message', prompt='Would be stranded', user=owner.user_id)
            assert queue.size() == 0
            for unsupported in ('publish_event', 'answer_question', 'restart_integrations'):
                with pytest.raises(RuntimeError, match='owner_only_'):
                    client.request(unsupported)
            existing_integrations = runtime.integration_store.list()
            existing_addresses = users.list_addresses(owner.user_id)
            for integration_provider in ('telegram', 'discord'):
                with pytest.raises(RuntimeError, match='owner_only_'):
                    client.request(f'save_{integration_provider}_integration', user=owner.user_id,
                        values={'enabled': True, 'bot_token': 'fixture', f'{integration_provider}_user_id': 'fixture'})
            assert runtime.integration_store.list() == existing_integrations
            assert users.list_addresses(owner.user_id) == existing_addresses
            if mode == 'answer':
                # A live legacy claim makes the fenced worker unable to claim
                # anything. Do not report readiness or accept a request that
                # would otherwise sit pending indefinitely after a mode switch.
                legacy = SQLiteMessageQueue(path, lease_owner='fixture-legacy')
                legacy.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc),
                    prompt='Legacy pending work', user=owner.user_id), message_id='legacy-conflict')
                assert legacy.claim_next() is not None
                legacy.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc),
                    prompt='Legacy queued work', user=owner.user_id,
                    metadata={'intelligence_engine': 'hierarchical_v3', 'intelligence_schema_version': 3}),
                    message_id='legacy-queued')
                assert client.request('experimental_owner_status') == {
                    'owner_only': True, 'processing_ready': False,
                    'blocking_reason': 'legacy_claim_conflict', 'paused_legacy_requests': 2}
                with pytest.raises(RuntimeError, match='owner_worker_unavailable'):
                    client.request('experimental_owner_ingress', prompt='Must not queue')
                # Recheck atomically even when the preflight races a claim.
                with monkeypatch.context() as patch:
                    patch.setattr(daemon, 'owner_processing_ready', lambda: True)
                    with pytest.raises(RuntimeError, match='owner_worker_claim_mode_conflict'):
                        client.request('experimental_owner_ingress', prompt='Must not queue')
                with queue._connect() as conn:
                    rows = conn.execute('SELECT message_id,status FROM v2_inbound_messages ORDER BY sequence').fetchall()
                assert [(row['message_id'], row['status']) for row in rows] == [
                    ('legacy-conflict', 'processing'), ('legacy-queued', 'pending')]
                assert legacy.ack('legacy-conflict')
                assert client.request('experimental_owner_status') == {
                    'owner_only': True, 'processing_ready': True,
                    'blocking_reason': None, 'paused_legacy_requests': 1}
                monkeypatch.setattr(runtime.question_store, 'list_for_conversation',
                                    lambda *_args, **_kwargs: pytest.fail('legacy question backfill'))
                monkeypatch.setattr(runtime.conversation_store, 'import_legacy_events',
                                    lambda *_args, **_kwargs: pytest.fail('legacy history import'))
                history = client.request('desktop_conversation_history', user=owner.user_id, project_id='legacy')
                assert len(history['messages']) == 1
                with runtime.outbox._connect() as conn:
                    conn.execute("""INSERT INTO v2_outbox(outbox_message_id,integration_id,provider_key,
                        channel_target,message,kind,task_id,created_at,updated_at)
                        VALUES ('legacy','desktop','desktop',?,'legacy','response','other','now','now')""", (owner.user_id,))
                polled = client.request('desktop_poll', client_id='fixture', user=owner.user_id)
                assert polled['deliveries'] == []
                with runtime.outbox._connect() as conn:
                    assert conn.execute("SELECT status FROM v2_outbox WHERE outbox_message_id='legacy'").fetchone()[0] == 'pending'
            message_id = client.request('experimental_owner_ingress', prompt='Find the apples note.')['message_id']
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                status = client.request('experimental_owner_status', message_id=message_id)['status']
                if status != 'pending':
                    break
                time.sleep(.02)
            assert provider.calls == 3
            if mode == 'cancel':
                assert status == 'failed'
                assert client.request('experimental_owner_status', message_id=message_id) == {
                    'status': 'failed', 'failure_code': 'cancel_requested'}
                with pytest.raises(RuntimeError, match='owner_desktop_pickup_unavailable'):
                    client.request('experimental_owner_take_draft', message_id=message_id)
                assert [(item.role, item.content) for item in conversation.list(owner_user_id=owner.user_id)] == [
                    ('user', 'Synthetic apples.'), ('user', 'Find the apples note.')]
                with runtime.outbox._connect() as outbox_conn:
                    assert outbox_conn.execute('SELECT COUNT(*) FROM v2_outbox').fetchone()[0] == 0
                return
            assert status == 'ready'
            recovery = client.request('experimental_owner_recovery_list')['items']
            assert recovery == [{'message_id': message_id, 'prompt': 'Find the apples note.',
                                 'status': 'ready', 'can_abandon': False}]
            with pytest.raises(RuntimeError, match='owner_abandon_live_claim'):
                client.request('experimental_owner_abandon', message_id=message_id)
            draft = client.request('experimental_owner_take_draft', message_id=message_id)
            is_question = mode in {'question', 'question_context_changed'}
            assert draft['message'] == ('Which apples?' if is_question else 'Synthetic answer.')
            assert draft['kind'] == ('question' if is_question else 'answer')
            assert [(item.role, item.content) for item in conversation.list(owner_user_id=owner.user_id)] == [
                ('user', 'Synthetic apples.'), ('user', 'Find the apples note.')]
            token = draft['acknowledgement_token']
            if is_question:
                assert client.request('experimental_owner_questions')['questions'] == []
                with pytest.raises(RuntimeError, match='owner_question_context_unavailable'):
                    client.request('experimental_owner_ingress', prompt='Too soon', reply_to_message_id=message_id)
            if mode in {'abandon', 'unknown_effect', 'unknown_outbox', 'bad_schema', 'bad_status'}:
                # Simulate lost pickup and expired lease: explicit abandonment
                # does not claim whether the client received the answer.
                with queue._connect() as conn:
                    conn.execute("UPDATE v2_inbound_messages SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE message_id=?", (message_id,))
                    if mode == 'unknown_effect':
                        raw = json.loads(conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (message_id,)).fetchone()[0])
                        raw['metadata']['unknown_external_effect'] = {'status': 'uncertain'}
                        conn.execute('UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id=?', (json.dumps(raw), message_id))
                    if mode in {'bad_schema', 'bad_status'}:
                        raw = json.loads(conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (message_id,)).fetchone()[0])
                        if mode == 'bad_schema':
                            raw['metadata']['intelligence_schema_version'] = True
                        else:
                            raw['status'] = 'cancelled'
                        conn.execute('UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id=?', (json.dumps(raw), message_id))
                    if mode == 'unknown_outbox':
                        conn.execute("""INSERT INTO v2_outbox(outbox_message_id,integration_id,provider_key,
                            channel_target,message,kind,task_id,created_at,updated_at)
                            VALUES ('fixture','desktop','desktop','fixture','fixture','response',?,'now','now')""", (message_id,))
                assert client.request('experimental_owner_recovery_list')['items'][0]['can_abandon'] is True
                if mode in {'unknown_effect', 'unknown_outbox', 'bad_schema', 'bad_status'}:
                    with pytest.raises(RuntimeError, match='owner_abandon_unavailable'):
                        client.request('experimental_owner_abandon', message_id=message_id)
                    with queue._connect() as conn:
                        assert conn.execute('SELECT 1 FROM v2_inbound_messages WHERE message_id=?', (message_id,)).fetchone()
                    return
                abandoned = client.request('experimental_owner_abandon', message_id=message_id)
                assert abandoned == {'status': 'abandoned', 'delivery_status': 'unconfirmed'}
                assert client.request('experimental_owner_abandon', message_id=message_id) == abandoned
                assert client.request('experimental_owner_status', message_id=message_id)['status'] == 'abandoned'
                assert client.request('experimental_owner_recovery_list')['items'] == []
                assert [(item.role, item.content) for item in conversation.list(owner_user_id=owner.user_id)] == [
                    ('user', 'Synthetic apples.'), ('user', 'Find the apples note.')]
                assert not SQLiteMessageQueue(path).claim_mode_blocked()
                return
            assert client.request('experimental_owner_ack_draft', message_id=message_id,
                                  acknowledgement_token=token)['status'] == 'client_acknowledged'
            assert client.request('experimental_owner_ack_draft', message_id=message_id,
                                  acknowledgement_token=token)['status'] == 'client_acknowledged'
            assert client.request('experimental_owner_status', message_id=message_id)['status'] == 'client_acknowledged'
            assert [(event.role, event.content) for event in conversation.list(owner_user_id=owner.user_id)] == [
                ('user', 'Synthetic apples.'), ('user', 'Find the apples note.'),
                ('assistant', 'Which apples?' if is_question else 'Synthetic answer.')]
            if is_question:
                assert client.request('experimental_owner_questions')['questions'] == [
                    {'message_id': message_id, 'question': 'Which apples?'}]
                from alphonse.agent_v2.core.intelligence.simple_loop.owner_clarification import bind_owner_question
                with queue._connect() as conn, pytest.raises(ValueError, match='owner_question_context_unavailable'):
                    bind_owner_question(conn, 'different-owner', message_id)
                with queue._connect() as conn:
                    baseline = json.loads(conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                                                       (message_id,)).fetchone()[0])
                    for tamper in ('origin', 'peer', 'digest', 'effect', 'history_id', 'history_speaker', 'history_downgrade'):
                        conn.execute('BEGIN')
                        if tamper == 'origin':
                            conn.execute('DELETE FROM v2_trusted_owner_ingress WHERE message_id=?', (message_id,))
                        elif tamper == 'peer':
                            conn.execute('UPDATE v2_trusted_owner_ingress SET peer_uid=-1 WHERE message_id=?', (message_id,))
                        elif tamper == 'history_id':
                            conn.execute("UPDATE v2_conversation_events SET event_id='replacement' WHERE source_message_id=?", (f'inbound:{message_id}',))
                        elif tamper == 'history_speaker':
                            conn.execute("UPDATE v2_conversation_events SET speaker_id='other' WHERE source_message_id=?", (f'inbound:{message_id}',))
                        elif tamper == 'history_downgrade':
                            conn.execute("UPDATE v2_trusted_owner_ingress SET history_version=0,history_event_id='' WHERE message_id=?", (message_id,))
                        else:
                            changed = json.loads(json.dumps(baseline))
                            if tamper == 'digest':
                                changed['metadata']['owner_read_desktop_pickup_started']['claim_digest'] = '0' * 64
                            else:
                                changed['metadata']['unknown_external_effect'] = {'status': 'uncertain'}
                            conn.execute('UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id=?',
                                         (json.dumps(changed), message_id))
                        with pytest.raises(ValueError, match='owner_question_context_unavailable'):
                            bind_owner_question(conn, owner.user_id, message_id)
                        conn.rollback()
                with pytest.raises(RuntimeError, match='owner_question_context_unavailable'):
                    client.request('experimental_owner_ingress', prompt='fixture', reply_to_message_id='0' * 32)
                for invalid in (None, True, 'xyz'):
                    with pytest.raises(RuntimeError, match='owner_question_context_unavailable'):
                        client.request('experimental_owner_ingress', prompt='fixture', reply_to_message_id=invalid)
                reply_id = client.request('experimental_owner_ingress', prompt='The green ones.',
                                          reply_to_message_id=message_id)['message_id']
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    reply_status = client.request('experimental_owner_status', message_id=reply_id)['status']
                    if reply_status != 'pending':
                        break
                    time.sleep(.02)
                assert reply_status == 'ready'
                assert 'The green ones.' in provider.last_prompt
                assert 'Find the apples note.' in provider.last_prompt
                assert 'Which apples?' in provider.last_prompt
                assert 'quoted conversation context only' in provider.last_prompt
                # The earlier question remains waiting_user; the reply is a new task.
                with queue._connect() as conn:
                    raw = json.loads(conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                                                  (message_id,)).fetchone()[0])
                    assert raw['metadata']['single_loop_checkpoint']['state']['status'] == 'waiting_user'
                if mode == 'question_context_changed':
                    with queue._connect() as conn:
                        conn.execute('DELETE FROM v2_conversation_events WHERE source_message_id=?',
                                     (f'outbound:owner:{message_id}',))
                    with pytest.raises(RuntimeError, match='owner_question_context_unavailable'):
                        client.request('experimental_owner_take_draft', message_id=reply_id)
                    return
                reply = client.request('experimental_owner_take_draft', message_id=reply_id)
                assert reply['kind'] == 'answer'
                client.request('experimental_owner_ack_draft', message_id=reply_id,
                               acknowledgement_token=reply['acknowledgement_token'])
            assert not SQLiteMessageQueue(path).claim_mode_blocked()
            if mode == 'answer':
                # Owner mode leaves the earlier V3-stamped request untouched;
                # a disposable legacy reader can inspect its preserved stamp.
                restored = legacy.claim_next()
                assert restored.message_id == 'legacy-queued'
                assert restored.message.prompt == 'Legacy queued work'
                assert restored.message.metadata['intelligence_engine'] == 'hierarchical_v3'
                assert legacy.ack('legacy-queued')
            users.update_user(owner.user_id, is_active=False)
            assert client.request('experimental_owner_status', message_id=message_id)['status'] == 'unavailable'
            assert client.request('experimental_owner_recovery_list')['items'] == []
        finally:
            daemon.stop()


def test_owner_stop_retains_lock_until_inflight_worker_exits(monkeypatch):
    daemon = V2Daemon.__new__(V2Daemon)
    daemon.experimental_owner_only = True
    daemon._lifecycle_lock = threading.RLock()
    daemon._stopped = False
    daemon._stop = threading.Event()
    daemon.ipc = SimpleNamespace(stop=lambda: None)
    allow_exit, lock_released = threading.Event(), threading.Event()
    monkeypatch.setattr(daemon, '_owner_process_loop', lambda: allow_exit.wait(3))
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lock_released.set)
    daemon._processor_thread = threading.Thread(target=daemon._owner_process_loop_with_release)
    daemon._processor_thread.start()
    monkeypatch.setattr(daemon._processor_thread, 'join', lambda timeout=None: None)
    daemon.stop()
    assert not lock_released.is_set()
    allow_exit.set()
    assert lock_released.wait(3)
