"""Replay a blocked Desktop read with disposable history and scripted inference."""
import json
import os
import time

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.simple_loop.owner_work_log import owner_work_log
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


PROMPT = ('¿Podemos retomar el diagnóstico del boiler sintético? '
          'Creo que ayer comprobaste si respondía a la red. ¿Qué sigue?')
RECOVERY = {
    'reply': ('Busqué en el historial, pero todavía no puedo confirmar qué se probó ayer ni si puedo controlar el boiler.\n\n'
              '¿Qué marca y modelo tiene, o con qué app lo manejas? Con ese dato podemos revisar las opciones de control '
              'y empezar por consultar su estado.'),
    'language': 'es',
    'activity': {'label': 'Ver actividad', 'detail': 'Busqué en el historial de la conversación.'},
}
USAGE = {'input_tokens': 500, 'output_tokens': 100}


def _wait(daemon, message_id):
    deadline = time.monotonic() + 8
    status = daemon.owner_message_status_payload(message_id)
    while status['status'] == 'pending' and time.monotonic() < deadline:
        time.sleep(.01)
        status = daemon.owner_message_status_payload(message_id)
    return status


@pytest.mark.parametrize('message_id', [[], {}, 'not-a-request-id'])
def test_owner_status_rejects_invalid_message_ids_before_recovery_lookup(message_id):
    daemon = V2Daemon.__new__(V2Daemon)
    assert daemon.owner_message_status_payload(message_id) == {'status': 'unavailable'}


@pytest.mark.parametrize('case', ['stays_blocked', 'other_language', 'model_tool', 'unknown_usage', 'cancelled', 'no_observation'])
def test_contextual_failure_is_a_scoped_recovery_reply_not_a_draft(tmp_path, monkeypatch, case):
    prompt = PROMPT
    copy = RECOVERY
    topic = 'boiler'
    if case == 'other_language':
        prompt = '昨日話した折り紙の続きをしたいです。どの折り方を選んだか覚えていますか？'
        copy = {'reply': '履歴を検索しましたが、昨日どの折り方を選んだか確認できませんでした。\n\n覚えている作品名はありますか？',
                'language': 'ja', 'activity': {'label': '作業内容を見る', 'detail': '会話の履歴を検索しました。'}}
        topic = '折り紙'

    class Provider:
        def __init__(self):
            self.work_calls = []
            self.recovery_calls = []

        def generate_json(self, request):
            if request.metadata.get('owner_failure_recovery') is True:
                self.recovery_calls.append(request)
                assert request.tools == ()
                assert 'tool_id' not in request.output_schema['properties']
                assert prompt in request.prompt
                assert 'Searched conversation history' in request.prompt
                assert 'PRIVATE_SOURCE_NOTE' not in request.prompt
                if case == 'cancelled':
                    assert daemon.kill_switch.request_cancel() is not None
                if case == 'model_tool':
                    return InferenceResult(json_value={**copy, 'kind': 'tool'}, usage=USAGE,
                                           tool_call={'tool_id': 'native.bash', 'arguments': {}})
                return InferenceResult(json_value=copy, usage={} if case == 'unknown_usage' else USAGE)
            self.work_calls.append(request)
            if len(self.work_calls) == 1 and case != 'no_observation':
                decision = {'kind': 'tool', 'tool_id': 'native.search_memory', 'text': None,
                            'arguments': {'query': topic, 'source_types': ['conversation']}}
            elif len(self.work_calls) <= 2:
                decision = {'kind': 'tool', 'tool_id': 'native.read_memory_source', 'text': None,
                            'arguments': {'source_id': 'inaccessible-synthetic-source'}}
            else:
                assert 'suggested_followup' in request.prompt
                assert 'Con ese dato podemos revisar' in request.prompt
                decision = {'kind': 'answer', 'tool_id': None, 'arguments': None,
                            'text': 'Gracias, con ese dato puedo orientar la comprobación.'}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision), usage=USAGE)

    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    conversation.record(owner_user_id=owner.user_id, role='user', source='desktop',
                        source_message_id='synthetic-earlier', content=f'PRIVATE_SOURCE_NOTE {topic}')
    provider = Provider()
    runtime = build_runtime_host(user_store=users, messages=queue,
        question_store=SQLiteQuestionStore(path), conversation_store=conversation,
        outbox=SQLiteOutboundStore(path),
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    daemon = V2Daemon(runtime, daemon_id='failure-recovery-fixture', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        message_id = queue.enqueue_trusted_local_owner(prompt=prompt, peer_uid=os.getuid())
        expected = {'status': 'failed', 'failure_code': 'cancel_requested' if case == 'cancelled' else 'source_unavailable'}
        if case in {'stays_blocked', 'other_language'}:
            expected['failure_recovery'] = copy
        assert _wait(daemon, message_id) == expected
        assert len(provider.recovery_calls) == (0 if case == 'no_observation' else 1)
        log = owner_work_log(queue, message_id)
        assert log['status'] == 'blocked'
        assert log['events'] == ([] if case == 'no_observation' else [
            {'label': 'Searched conversation history', 'outcome': 'Observation recorded'},
        ])
        with pytest.raises(ValueError, match='owner_desktop_pickup_unavailable'):
            daemon.take_owner_desktop_draft(message_id)
        history = daemon.desktop_conversation_history(user=owner.user_id)
        row = next(item for item in history if item['id'] == message_id)
        assert row['role'] == 'user'
        assert row.get('failure_recovery') == expected.get('failure_recovery')
        assert not any(item['role'] == 'assistant' for item in history)
        with queue._connect() as conn:
            raw = json.loads(conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                                          (message_id,)).fetchone()[0])
        envelope = raw['metadata']['single_loop_checkpoint']
        final_calls = [entry for entry in envelope['budget']['reservations'].values() if entry['lane'] == 'final']
        assert len(final_calls) == len(provider.recovery_calls)
        assert {entry['status'] for entry in final_calls} <= {'settled'}
        assert all(entry['outcome'] == ('success' if 'failure_recovery' in expected else 'failure') for entry in final_calls)
        assert envelope['state']['response'] == ''
        if case == 'stays_blocked':
            followup = queue.enqueue_trusted_local_owner(prompt='Lo manejo con la app Sintética.', peer_uid=os.getuid())
            assert _wait(daemon, followup) == {'status': 'ready'}
    finally:
        daemon.stop()

    if case == 'stays_blocked':
        restarted = V2Daemon(runtime, daemon_id='failure-recovery-restart', experimental_owner_only=True)
        assert restarted.owner_message_status_payload(message_id) == expected
        with queue._connect() as conn:
            conn.execute('UPDATE v2_users SET is_active=0 WHERE user_id=?', (owner.user_id,))
        assert restarted.owner_message_status_payload(message_id) == {'status': 'unavailable'}
