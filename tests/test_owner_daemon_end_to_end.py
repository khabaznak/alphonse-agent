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
    assert _owner_public_failure_code('candidate_retry_deferred') == 'source_unavailable'
    assert _owner_public_failure_code('candidate_deadline_exceeded') == 'time_limit'
    assert _owner_public_failure_code('candidate_model_usage_overrun') == 'usage_limit_exceeded'
    assert _owner_public_failure_code('BudgetExhausted:budget_exhausted:actions') == 'action_limit'
    assert _owner_public_failure_code('candidate_model_usage_unknown_or_overrun') == 'usage_unconfirmed'
    assert _owner_public_failure_code('candidate_question_not_verified') == 'clarification_unverified'
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
    monkeypatch.setattr(module, '_trusted_local_bash_enabled', lambda: False)
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
            self.runtime = runtime
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
    if mode in {'question', 'question_context_changed'}:
        # This fixture tests IPC correlation, not Jev's semantic judgment.
        runtime.core.system_one = SimpleNamespace(review_clarification_effect_claim=lambda **_: True)
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
        assert context.identity_resolver is runtime.identity_resolver
        assert context.project_store is runtime.project_store
        execution = context.tool_execution_context(task)
        assert execution.skill_store is context.skill_store
        assert execution.artifact_resources is context.artifact_resources
        assert execution.skill_bindings is context.skill_bindings
        assert execution.identity_resolver is context.identity_resolver
        assert execution.project_store is context.project_store
        # Native Bash uses this actor-bound workspace when a conversation has
        # no project or explicit cwd. Supplying the dependencies alone does
        # not advertise or authorize Bash in the owner worker.
        from alphonse.agent_v2.core.tools.registry.native.bash import _resolve_cwd
        assert _resolve_cwd(None, context=execution) == str(users.ordinary_workspace(owner.user_id))
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
                # Expiry alone is insufficient: unknown effects, outbox state
                # and malformed checkpoints must not be advertised as safe.
                assert client.request('experimental_owner_recovery_list')['items'][0]['can_abandon'] is (mode == 'abandon')
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
                # A second distinct owner reply may refer to the same prior
                # question. Both are real, separately recorded owner ingress.
                second_reply_id = client.request('experimental_owner_ingress', prompt='The red ones.',
                                                 reply_to_message_id=message_id)['message_id']
                # The exchange snapshot preserves the exact separate turns;
                # it must not silently trust a different reply or altered
                # earlier history. It does not itself grant action authority.
                from alphonse.agent_v2.core.intelligence.simple_loop.owner_clarification import load_bound_owner_exchange
                with queue._connect() as conn:
                    conn.execute('BEGIN')
                    queued = conn.execute('SELECT prompt,metadata_json FROM v2_inbound_messages WHERE message_id=?',
                                          (reply_id,)).fetchone()
                    task = SimpleNamespace(user=owner.user_id, message_id=reply_id, goal=queued['prompt'],
                                           metadata=json.loads(queued['metadata_json']))
                    exchange = load_bound_owner_exchange(conn, task)
                    assert (exchange.prior_user_message, exchange.assistant_question, exchange.owner_reply) == (
                        'Find the apples note.', 'Which apples?', 'The green ones.')
                    assert exchange.prior_message_id == message_id and exchange.reply_message_id == reply_id
                    assert len(exchange.sha256) == 64
                    task.goal = 'A different reply.'
                    with pytest.raises(ValueError, match='owner_question_context_unavailable'):
                        load_bound_owner_exchange(conn, task)
                    task.goal = queued['prompt']
                    task.user = 'different-owner'
                    with pytest.raises(ValueError, match='owner_question_context_unavailable'):
                        load_bound_owner_exchange(conn, task)
                    task.user = owner.user_id

                    # A disconnected proposal prototype may bind these exact
                    # turns, but only a separately authenticated UI event may
                    # confirm its immutable fields. No natural-language reply
                    # or model judgment is interpreted as confirmation here.
                    from datetime import timedelta
                    from alphonse.agent_v2.core.intelligence.simple_loop.owner_reminder_proposal import (
                        ProposalUnavailable, prepare_locked, confirm_locked, cancel_locked, consume_locked)
                    now = datetime.now(timezone.utc)
                    proposal = prepare_locked(conn, task, message='check the apples',
                        due_at=(now + timedelta(minutes=5)).isoformat(), timezone_name='UTC', now=now)
                    second_row = conn.execute('SELECT prompt,metadata_json FROM v2_inbound_messages WHERE message_id=?',
                                              (second_reply_id,)).fetchone()
                    second_task = SimpleNamespace(user=owner.user_id, message_id=second_reply_id,
                        goal=second_row['prompt'], metadata=json.loads(second_row['metadata_json']))
                    second_proposal = prepare_locked(conn, second_task, message='check the red apples',
                        due_at=(now + timedelta(minutes=5)).isoformat(), timezone_name='UTC', now=now)
                    with pytest.raises(ProposalUnavailable):
                        consume_locked(conn, task, proposal_id=proposal['id'], now=now, is_cancelled=lambda: False)
                    with pytest.raises(ProposalUnavailable):
                        confirm_locked(conn, task, proposal_id=proposal['id'], proposal_sha256=proposal['proposal_sha256'],
                                       authenticated_owner='different-owner', now=now)
                    with pytest.raises(ProposalUnavailable):
                        confirm_locked(conn, task, proposal_id=proposal['id'], proposal_sha256='0' * 64,
                                       authenticated_owner=owner.user_id, now=now)
                    conn.execute('SAVEPOINT altered_proposal')
                    conn.execute("UPDATE owner_reminder_proposals SET message='different content' WHERE id=?", (proposal['id'],))
                    with pytest.raises(ProposalUnavailable):
                        confirm_locked(conn, task, proposal_id=proposal['id'], proposal_sha256=proposal['proposal_sha256'],
                                       authenticated_owner=owner.user_id, now=now)
                    conn.execute('ROLLBACK TO altered_proposal')
                    conn.execute('RELEASE altered_proposal')
                    confirm_locked(conn, task, proposal_id=proposal['id'], proposal_sha256=proposal['proposal_sha256'],
                                   authenticated_owner=owner.user_id, now=now)
                    confirm_locked(conn, second_task, proposal_id=second_proposal['id'],
                                   proposal_sha256=second_proposal['proposal_sha256'],
                                   authenticated_owner=owner.user_id, now=now)
                    conn.execute('SAVEPOINT cancellation')
                    cancel_locked(conn, task, proposal_id=proposal['id'], authenticated_owner=owner.user_id, now=now)
                    with pytest.raises(ProposalUnavailable):
                        consume_locked(conn, task, proposal_id=proposal['id'], now=now, is_cancelled=lambda: False)
                    conn.execute('ROLLBACK TO cancellation')
                    conn.execute('RELEASE cancellation')
                    with pytest.raises(ProposalUnavailable):
                        consume_locked(conn, task, proposal_id=proposal['id'], now=now, is_cancelled=lambda: True)
                    first = consume_locked(conn, task, proposal_id=proposal['id'], now=now, is_cancelled=lambda: False)
                    second = consume_locked(conn, task, proposal_id=proposal['id'], now=now, is_cancelled=lambda: False)
                    assert first['status'] == 'created' and second['status'] == 'already_consumed'
                    assert first['id'] == second['id']
                    with pytest.raises(ProposalUnavailable, match='prior_intent_already_consumed'):
                        consume_locked(conn, second_task, proposal_id=second_proposal['id'], now=now,
                                       is_cancelled=lambda: False)
                    assert conn.execute('SELECT COUNT(*) FROM owner_desktop_reminders WHERE owner_id=?',
                                        (owner.user_id,)).fetchone()[0] == 1

                    conn.execute('DELETE FROM v2_conversation_events WHERE source_message_id=?',
                                 (f'outbound:owner:{message_id}',))
                    with pytest.raises(ValueError, match='owner_question_context_unavailable'):
                        load_bound_owner_exchange(conn, task)
                    with pytest.raises(ValueError, match='owner_question_context_unavailable'):
                        consume_locked(conn, task, proposal_id=proposal['id'], now=now, is_cancelled=lambda: False)
                    conn.rollback()
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
    daemon.ipc = SimpleNamespace(stop=lambda: None, wait_quiescent=lambda timeout=None: True)
    allow_exit, lock_released = threading.Event(), threading.Event()
    monkeypatch.setattr(daemon, '_owner_process_loop', lambda: allow_exit.wait(3))
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lock_released.set)
    daemon._processor_thread = threading.Thread(target=daemon._owner_process_loop_with_release)
    daemon._processor_thread.start()
    real_join = daemon._processor_thread.join
    monkeypatch.setattr(daemon._processor_thread, 'join', lambda timeout=None: None if timeout is not None else real_join())
    daemon.stop()
    assert not lock_released.is_set()
    allow_exit.set()
    assert lock_released.wait(3)


def test_owner_stop_retains_lock_until_ipc_requests_finish(monkeypatch):
    daemon = V2Daemon.__new__(V2Daemon)
    daemon.experimental_owner_only = True
    daemon._lifecycle_lock = threading.RLock()
    daemon._stopped = False
    daemon._stop = threading.Event()
    daemon._processor_thread = None
    drained, released = threading.Event(), threading.Event()
    daemon.ipc = SimpleNamespace(stop=lambda: None, wait_quiescent=lambda timeout=None: drained.wait(timeout))
    monkeypatch.setattr(daemon, '_release_single_instance_lock', released.set)
    daemon.stop()
    assert not released.is_set()
    drained.set()
    assert released.wait(3)


def test_ipc_stop_tracks_an_accepted_request(monkeypatch):
    import socket
    temp_dir = tempfile.TemporaryDirectory(prefix='ipc-drain-', dir='/private/tmp')
    server = V2DaemonServer(SimpleNamespace(), os.path.join(temp_dir.name, 'p.sock'))
    entered, finish = threading.Event(), threading.Event()
    def handle(connection):
        with connection:
            entered.set()
            assert finish.wait(3)
    monkeypatch.setattr(server, '_handle', handle)
    server.start()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(str(server.socket_path))
        assert entered.wait(3)
        server.stop()
        assert not server.wait_quiescent(0)
        finish.set()
        assert server.wait_quiescent(3)
    finally:
        finish.set()
        client.close()
        server.stop()
        temp_dir.cleanup()


def test_concurrent_start_creates_only_one_owner_worker(tmp_path, monkeypatch):
    """Two starts on the same object must not share one lock and start two workers."""
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_processor import OwnerReadConversationProcessor

    daemon = V2Daemon.__new__(V2Daemon)
    daemon.experimental_owner_only = True
    processor = OwnerReadConversationProcessor.__new__(OwnerReadConversationProcessor)
    processor.atomic_checkpoints = object()
    daemon.runtime = SimpleNamespace(
        queue=SQLiteMessageQueue(tmp_path / 'queue.sqlite', experimental_claim_fencing=True),
        processor=SimpleNamespace(owner_read_candidate=processor))
    daemon._lifecycle_lock = threading.RLock()
    daemon._stopped = False
    daemon._processor_thread = None
    daemon._stop = threading.Event()
    entered, release, worker_started, worker_exit, second_attempt, duplicate_ipc = (
        threading.Event() for _ in range(6))
    calls = []

    class IPC:
        def start(self):
            calls.append('ipc')
            if len(calls) > 1:
                duplicate_ipc.set()
            entered.set()
            assert release.wait(3)

        def stop(self):
            pass
        def wait_quiescent(self, timeout=None):
            return True

    daemon.ipc = IPC()
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    def worker():
        calls.append('worker')
        worker_started.set()
        assert worker_exit.wait(3)
    monkeypatch.setattr(daemon, '_owner_process_loop_with_release', worker)
    failures = []
    def start(second=False):
        try:
            if second:
                second_attempt.set()
            daemon.start()
        except Exception as exc:
            failures.append(exc)

    first = threading.Thread(target=start)
    second = threading.Thread(target=start, args=(True,))
    first.start()
    assert entered.wait(3)
    second.start()
    try:
        assert second_attempt.wait(3)
        assert not duplicate_ipc.wait(0.2)
        release.set()
        first.join(3)
        second.join(3)
        assert not first.is_alive() and not second.is_alive()
        assert worker_started.wait(3)
        assert not failures
        assert calls.count('ipc') == 1
        assert calls.count('worker') == 1
    finally:
        release.set()
        worker_exit.set()
        daemon.stop()


@pytest.mark.parametrize('failure_stage', ['ipc', 'worker'])
def test_owner_start_failure_stops_outside_lifecycle_lock(tmp_path, monkeypatch, failure_stage):
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_processor import OwnerReadConversationProcessor

    daemon = V2Daemon.__new__(V2Daemon)
    daemon.experimental_owner_only = True
    processor = OwnerReadConversationProcessor.__new__(OwnerReadConversationProcessor)
    processor.atomic_checkpoints = object()
    daemon.runtime = SimpleNamespace(
        queue=SQLiteMessageQueue(tmp_path / 'queue.sqlite', experimental_claim_fencing=True),
        processor=SimpleNamespace(owner_read_candidate=processor))
    daemon._lifecycle_lock = threading.RLock()
    daemon._stopped = False
    daemon._processor_thread = None
    daemon._stop = threading.Event()
    cleanup_acquired = threading.Event()
    class IPC:
        def start(self):
            if failure_stage == 'ipc':
                raise RuntimeError('synthetic startup failure')
        def stop(self):
            def cleanup():
                with daemon._lifecycle_lock:
                    cleanup_acquired.set()
            thread = threading.Thread(target=cleanup)
            thread.start()
            thread.join(1)
        def wait_quiescent(self, timeout=None):
            return True
    daemon.ipc = IPC()
    released = []
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: released.append(True))
    original_start = threading.Thread.start
    def start_thread(thread):
        if failure_stage == 'worker' and thread.name == 'alphonse-owner-read':
            raise RuntimeError('synthetic startup failure')
        return original_start(thread)
    monkeypatch.setattr(threading.Thread, 'start', start_thread)
    with pytest.raises(RuntimeError, match='synthetic startup failure'):
        daemon.start()
    assert cleanup_acquired.is_set()
    assert released == [True]
    with pytest.raises(RuntimeError, match='daemon_stopped'):
        daemon.start()


def test_owner_stop_during_start_waits_then_stops_worker(tmp_path, monkeypatch):
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_processor import OwnerReadConversationProcessor

    daemon = V2Daemon.__new__(V2Daemon)
    daemon.experimental_owner_only = True
    processor = OwnerReadConversationProcessor.__new__(OwnerReadConversationProcessor)
    processor.atomic_checkpoints = object()
    daemon.runtime = SimpleNamespace(
        queue=SQLiteMessageQueue(tmp_path / 'queue.sqlite', experimental_claim_fencing=True),
        processor=SimpleNamespace(owner_read_candidate=processor))
    daemon._lifecycle_lock = threading.RLock()
    daemon._stopped = False
    daemon._processor_thread = None
    daemon._stop = threading.Event()
    entered, release, stopped = (threading.Event() for _ in range(3))
    class IPC:
        def start(self):
            entered.set()
            assert release.wait(3)
        def stop(self):
            stopped.set()
        def wait_quiescent(self, timeout=None):
            return True
    daemon.ipc = IPC()
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_owner_process_loop_with_release', lambda: daemon._stop.wait(3))
    starter = threading.Thread(target=daemon.start)
    stopper = threading.Thread(target=daemon.stop)
    starter.start()
    assert entered.wait(3)
    stopper.start()
    try:
        assert not stopped.wait(0.2)
        release.set()
        starter.join(3)
        stopper.join(3)
        assert not starter.is_alive() and not stopper.is_alive()
        assert stopped.is_set() and daemon._stopped
        assert daemon._processor_thread is not None and not daemon._processor_thread.is_alive()
    finally:
        release.set()
        daemon._stop.set()
