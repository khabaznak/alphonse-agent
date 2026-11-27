"""Drive disposable Desktop requests through the working and final call lanes."""
import json
import os
import time
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.openai_codex import _json_envelope
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.simple_loop.router_candidate_chooser import RouterCandidateChooser
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import ToolDefinition
from alphonse.agent_v2.core.tools.registry.native.web import WEB_SEARCH_ARGUMENT_SCHEMA, WEB_SEARCH_TOOL_ID
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


@pytest.mark.parametrize('final_usage,show_report', [
    pytest.param(None, False, id='answers-before-limit'),
    pytest.param({'input_tokens': 16392, 'output_tokens': 2056}, True, id='fits-task-over-forecast'),
    pytest.param({'input_tokens': 786432 - 11 * 500 + 1, 'output_tokens': 100}, False, id='over-task-input'),
    pytest.param({'input_tokens': 16392, 'output_tokens': 196608 - 11 * 100 + 1}, False, id='over-task-output'),
])
def test_context_compaction_replan_and_reserved_report(tmp_path, monkeypatch, final_usage, show_report):
    stop_at_limit = final_usage is not None
    prompt = 'Busca qué muestra el catálogo público sobre la muestra sintética.'
    finding = 'El catálogo indica que la muestra es azul.'
    retained_marker = 'middle of the original observation'
    report = {'reply': f'Alcancé el límite de llamadas al modelo. {finding}', 'language': 'es'}
    work, final, searches, assessments, rejected_sizes = [], [], [], [], []

    prepare = RouterCandidateChooser.prepare

    def capture_preflight(self, projection, output_limit):
        try:
            return prepare(self, projection, output_limit)
        except ValueError as exc:
            if str(exc) == 'candidate_router_input_exceeds_reservation':
                rejected_sizes.append(len(projection.encode('utf-8')))
            raise

    monkeypatch.setattr(RouterCandidateChooser, 'prepare', capture_preflight)

    class Model:
        def generate_json(self, request):
            assert len(_json_envelope(request).encode('utf-8')) <= 65536
            if request.metadata.get('owner_failure_recovery') is True:
                final.append(request)
                assert request.tools == ()
                assert set(request.output_schema['properties']) == {'reply', 'language'}
                assert finding in request.prompt and 'working model call limit' in request.prompt
                value = report
                usage = final_usage
            else:
                work.append(request)
                index = len(work)
                assert ('Reassess the approach' in request.prompt) == (index == 4)
                if index == 6:
                    assert 'observation excerpted' in request.prompt
                    assert finding in request.prompt
                    assert retained_marker not in request.prompt
                if index == 6 and not stop_at_limit:
                    value = {'kind': 'answer', 'text': finding, 'tool_id': None, 'arguments': None}
                else:
                    prefix = 'Look for a different catalog' if index >= 4 else 'Look for a product name'
                    value = {'kind': 'tool', 'text': None, 'tool_id': WEB_SEARCH_TOOL_ID,
                             'arguments': {'query': f'{prefix} {index}'}}
                usage = {'input_tokens': 500, 'output_tokens': 100}
            return InferenceResult(json_value=value, raw_response=json.dumps(value), usage=usage)

    class Jev:
        settings = SimpleNamespace(enabled=True, no_threshold=.2, yes_threshold=.8)

        def judge_observed_progress(self, *, state):
            assert state['request'] == prompt
            assert len(state['observations']) == 3
            assessments.append(state)
            if not state['prior_observations']:
                return {'new_fact': .04, 'advances_question': .03}
            return {'new_fact': .95, 'advances_question': .96}

    def search(args):
        searches.append(args['query'])
        if len(searches) == 1:
            return {'output': None, 'exception': {'code': 'searxng_timeout', 'message': 'Synthetic timeout',
                                                   'retryable': True}}
        results = []
        if 4 <= len(searches) <= 5:
            large_text = finding + (' x' * 13000) + retained_marker + (' y' * 1700)
            results = [{'title': finding, 'content': large_text}]
        return {'query': args['query'], 'provider': 'searxng', 'result_count': len(results),
                'results': results, 'answers': [], 'suggestions': []}

    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    runtime = build_runtime_host(user_store=users, messages=queue,
        question_store=SQLiteQuestionStore(path), conversation_store=SQLiteConversationStore(path),
        outbox=SQLiteOutboundStore(path),
        inference=InferenceRouter(provider=Model(), default_profile=ModelProfile('fixture', 'fixture', 'offline')),
        experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    runtime.core.system_one = Jev()
    runtime.core.tools.register(ToolDefinition(
        ToolDescriptor(WEB_SEARCH_TOOL_ID, 'web_search', ToolKind.NATIVE, description='Search the public catalog.',
                       argument_schema=WEB_SEARCH_ARGUMENT_SCHEMA, read_only=True),
        search, WEB_SEARCH_ARGUMENT_SCHEMA))
    daemon = V2Daemon(runtime, daemon_id='context-limit-fixture', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        message_id = queue.enqueue_trusted_local_owner(prompt=prompt, peer_uid=os.getuid())
        deadline = time.monotonic() + 10
        status = daemon.owner_message_status_payload(message_id)
        while status['status'] == 'pending' and time.monotonic() < deadline:
            time.sleep(.02)
            status = daemon.owner_message_status_payload(message_id)
        if stop_at_limit:
            expected = {'status': 'failed', 'failure_code': 'model_call_limit'}
            if show_report:
                expected['failure_recovery'] = report
            assert status == expected
            history = daemon.desktop_conversation_history(user=owner.user_id)
            message = next(item for item in history if item['id'] == message_id)
            assert message.get('failure_recovery') == expected.get('failure_recovery')
            if show_report:
                assert message['failure_code'] == 'model_call_limit'
        else:
            assert status == {'status': 'ready'}
            assert daemon.take_owner_desktop_draft(message_id)['message'] == finding
    finally:
        daemon.stop()

    assert len(work) == (11 if stop_at_limit else 6)
    assert len(searches) == (10 if stop_at_limit else 5)
    assert len(assessments) == (3 if stop_at_limit else 1)
    assert len(final) == int(stop_at_limit)
    assert rejected_sizes and max(rejected_sizes) < 65536
    with queue._connect() as conn:
        checkpoint = json.loads(conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                                             (message_id,)).fetchone()[0])['metadata']
    assert retained_marker in json.dumps(checkpoint['owner_read_observations'])
    envelope = checkpoint['single_loop_checkpoint']
    assert envelope['state']['progress_reviewed_count'] == (9 if stop_at_limit else 3)
    reservations = tuple(envelope['budget']['reservations'].values())
    assert not any(row['unknown_tokens'] for row in reservations)
    assert sum(row['kind'] == 'model' for row in reservations) == len(work) + len(final)
    if stop_at_limit:
        final_reservation, = [row for row in reservations if row['lane'] == 'final']
        assert final_reservation['outcome'] == ('success' if show_report else 'failure')
        assert {key: final_reservation['actual'][key] for key in final_usage} == final_usage
