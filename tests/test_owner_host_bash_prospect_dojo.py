"""Exercise the native owner Bash path using only a disposable scripted fixture.

The scripted actions are a plumbing test. They do not evaluate model judgment,
and no model-composed command is allowed to run on the host here.
"""
import json
import shlex
import time

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native.memory import (
    build_read_memory_source_tool_definition,
    build_search_memory_tool_definition,
)
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.evaluation import dojo_lead_followup
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def test_scripted_prospect_obstacle_uses_owner_host_bash_and_independent_readback(tmp_path, monkeypatch):
    path = tmp_path / 'synthetic.sqlite'
    users = V2UserStore(path)
    users.set_workspace_root(tmp_path / 'workspace')
    owner = users.onboard(display_name='Synthetic owner')
    workspace = dojo_lead_followup.seed(users.ordinary_workspace(owner.user_id))
    program = '''import csv,pathlib
p=pathlib.Path('prospect-data')
with (p/'leads.csv').open(newline='') as f: leads=list(csv.DictReader(f))
with (p/'contact-history.csv').open(newline='') as f: contacts=list(csv.DictReader(f))
recent={r['lead_id'] for r in contacts if r['contacted_on'] >= '2026-11-01'}
with (p/'follow-up.csv').open('w',newline='') as f:
 w=csv.writer(f,lineterminator='\\n'); w.writerow(['name','company'])
 w.writerows((r['name'],r['company']) for r in leads if r['status']=='qualified' and r['lead_id'] not in recent)
'''

    class ScriptedProvider:
        supports_image_input = False
        calls = 0
        def generate_json(self, request):
            self.calls += 1
            if self.calls == 1:
                decision = {'kind': 'tool', 'tool_id': 'native.bash',
                            'arguments': {'command': 'python3 -c ' + shlex.quote(program)}, 'text': None}
            elif self.calls == 2:
                decision = {'kind': 'tool', 'tool_id': 'native.bash',
                            'arguments': {'command': 'cat prospect-data/follow-up.csv'}, 'text': None}
            else:
                assert 'Beatriz Cruz' in request.prompt and 'Devin Shaw' in request.prompt
                decision = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'The list is ready.'}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    provider = ScriptedProvider()
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    registry = InMemoryToolRegistry()
    for definition in (build_search_memory_tool_definition(), build_read_memory_source_tool_definition(),
                       build_bash_tool_definition()):
        registry.register(definition)
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(path),
        conversation_store=conversation, outbox=SQLiteOutboundStore(path), tools=registry,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'scripted', 'dojo-owner')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    runtime.processor.owner_read_candidate.enable_owner_host_bash(allow_unprotected_for_testing=True)
    daemon = V2Daemon(runtime, daemon_id='scripted-host-dojo', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        import os
        message_id = queue.enqueue_trusted_local_owner(prompt=dojo_lead_followup.goal(workspace), peer_uid=os.getuid())
        deadline = time.monotonic() + 10
        status = 'pending'
        while time.monotonic() < deadline:
            status = daemon.owner_message_status(message_id)
            if status not in {'pending', 'processing'}:
                break
            time.sleep(.01)
        assert status == 'ready'
        actual = dojo_lead_followup.readback(workspace)
        assert dojo_lead_followup.grade(actual)['effect_passed'] is True
        draft = daemon.take_owner_desktop_draft(message_id)
        assert draft['message'] == 'The list is ready.'
        daemon.acknowledge_owner_desktop_draft(message_id, draft['acknowledgement_token'])
        assert provider.calls == 3
    finally:
        daemon.stop()
