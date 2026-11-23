"""A real owner worker can finish after more than five ordinary tool calls."""
import json
import os
import time
import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


@pytest.mark.parametrize('requested,expected_status,observed', [(6, 'ready', 6), (11, 'failed', 10)])
def test_owner_worker_handles_long_tasks_and_still_enforces_limit(tmp_path, monkeypatch, requested, expected_status, observed):
    class Provider:
        supports_image_input = False
        calls = 0
        def generate_json(self, request):
            self.calls += 1
            decision = ({'kind': 'tool', 'tool_id': 'native.owner_clock', 'arguments': {}, 'text': None}
                        if self.calls <= requested else
                        {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'I observed the clock.'})
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    users.set_workspace_root(tmp_path / 'workspace')
    users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    provider = Provider()
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(path),
        conversation_store=conversation, outbox=SQLiteOutboundStore(path),
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    daemon = V2Daemon(runtime, daemon_id='budget-fixture', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        message = queue.enqueue_trusted_local_owner(prompt='Report the current clock.', peer_uid=os.getuid())
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and daemon.owner_message_status(message) in {'pending', 'processing'}:
            time.sleep(.01)
        assert daemon.owner_message_status(message) == expected_status
        assert provider.calls == observed + 1
        with queue._connect() as conn:
            raw = json.loads(conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                                          (message,)).fetchone()[0])
        assert len(raw['metadata']['owner_read_observations']) == observed
        if expected_status == 'ready':
            draft = daemon.take_owner_desktop_draft(message)
            assert 'I observed the clock.' in draft['message']
            daemon.acknowledge_owner_desktop_draft(message, draft['acknowledgement_token'])
        else:
            assert daemon.owner_message_status_payload(message)['failure_code'] == 'action_limit'
    finally:
        daemon.stop()
