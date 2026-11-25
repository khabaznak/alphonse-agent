"""Safe diagnosis of recorded owner failures; no raw provider text is exposed."""
import json
import os
import time
from dataclasses import replace
import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.simple_loop.owner_failure_diagnostics import public_failure_code, recent_owner_failures
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def test_only_exact_safe_failure_categories_are_public():
    assert public_failure_code('BudgetExhausted:budget_elapsed_exhausted') == 'time_limit'
    assert public_failure_code('candidate_model_timeout') == 'model_timeout'
    assert public_failure_code('candidate_model_timeout: private text') == 'needs_attention'
    assert public_failure_code('candidate_model_timeout', cancelled=True) == 'cancel_requested'


@pytest.mark.parametrize(('failure', 'code', 'description'), [
    ('openai_codex_timeout', 'model_timeout', 'The model provider timed out.'),
    ('private provider payload', 'model_unavailable', 'The model step failed; its specific cause was not recorded.'),
])
def test_owner_followup_sees_prior_safe_failure_without_private_provider_text(tmp_path, monkeypatch, failure, code, description):
    class Provider:
        supports_image_input = False
        def __init__(self):
            self.prompts = []
        def generate_json(self, request):
            self.prompts.append(request.prompt)
            if len(self.prompts) == 1:
                raise ValueError(failure)
            decision = {'kind': 'answer', 'tool_id': None, 'arguments': None,
                        'text': 'The previous model step timed out.'}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    provider = Provider()
    runtime = build_runtime_host(user_store=users, messages=queue,
        question_store=SQLiteQuestionStore(path), conversation_store=conversation,
        outbox=SQLiteOutboundStore(path),
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    daemon = V2Daemon(runtime, daemon_id='diagnostics-fixture', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        first = queue.enqueue_trusted_local_owner(prompt='Synthetic first request', peer_uid=os.getuid())
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and daemon.owner_message_status(first) in {'pending', 'processing'}:
            time.sleep(.01)
        assert daemon.owner_message_status_payload(first) == {'status': 'failed', 'failure_code': code}
        second = queue.enqueue_trusted_local_owner(prompt='What happened to the previous request?', peer_uid=os.getuid())
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and daemon.owner_message_status(second) in {'pending', 'processing'}:
            time.sleep(.01)
        assert daemon.owner_message_status(second) == 'ready'
        assert description in provider.prompts[-1]
        assert 'not established by this diagnostic' in provider.prompts[-1]
        assert failure not in provider.prompts[-1]
        assert '"failure_code"' not in provider.prompts[-1]
        events = conversation.list_before_source(owner_user_id=owner.user_id, project_id='',
                                                 source_message_id='inbound:' + second, limit=16)
        assert recent_owner_failures(queue, events, actor=owner.user_id, project_id='', current_message_id=second)[0]['description'] == description
        assert recent_owner_failures(queue, events, actor='other-owner', project_id='', current_message_id=second) == []
        assert recent_owner_failures(queue, events, actor=owner.user_id, project_id='other-project', current_message_id=second) == []
        assert recent_owner_failures(queue, [replace(events[0], event_id='forged')], actor=owner.user_id,
                                     project_id='', current_message_id=second) == []
        telegram_id = 'f' * 32
        conversation.record(owner_user_id=owner.user_id, role='user', content='Synthetic message',
            source='telegram', source_message_id='inbound:' + telegram_id)
        assert recent_owner_failures(queue, events, actor=owner.user_id, project_id='', current_message_id=telegram_id) == []
        with queue._connect() as conn:
            conn.execute('UPDATE v2_conversation_events SET content=? WHERE event_id=?', ('changed', events[0].event_id))
        assert recent_owner_failures(queue, events, actor=owner.user_id, project_id='', current_message_id=second) == []
    finally:
        daemon.stop()
