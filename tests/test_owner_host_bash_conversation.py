"""Trusted-host adapter exercised only against disposable temporary data."""
import json
import os
import tempfile
import time
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native.memory import build_search_memory_tool_definition, build_read_memory_source_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


@pytest.mark.parametrize('through_ipc', [False, True])
@pytest.mark.parametrize('trusted_local', [False, True])
def test_opt_in_owner_can_use_host_process_and_read_back_without_default_exposure(tmp_path, monkeypatch, through_ipc, trusted_local):
    class Provider:
        supports_image_input = False
        calls = 0
        def generate_json(self, request):
            self.calls += 1
            if self.calls == 1:
                decision = {'kind': 'tool', 'tool_id': 'native.bash',
                            'arguments': {'command': 'cat not-yet-created.txt'}, 'text': None}
            elif self.calls == 2:
                assert 'exit_code' in request.prompt and 'not-yet-created.txt' in request.prompt
                decision = {'kind': 'tool', 'tool_id': 'native.bash',
                            'arguments': {'command': "printf 'synthetic lead\\n' > prospects.txt"}, 'text': None}
            elif self.calls == 3:
                # The prior write emitted no path in stdout. The new decision
                # must still see the actual command that chose its target.
                assert 'prospects.txt' in request.prompt
                decision = {'kind': 'tool', 'tool_id': 'native.bash',
                            'arguments': {'command': 'cat prospects.txt'}, 'text': None}
            elif self.calls == 5:
                assert 'The note contains synthetic lead.' in request.prompt
                decision = {'kind': 'tool', 'tool_id': 'native.bash',
                            'arguments': {'command': 'cat prospects.txt'}, 'text': None}
            else:
                assert 'synthetic lead' in request.prompt
                assert 'unverified' in request.prompt
                decision = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'The note contains synthetic lead.'}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    users.set_workspace_root(tmp_path / 'workspace')
    owner = users.onboard(display_name='Synthetic owner')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    registry = InMemoryToolRegistry()
    for definition in (build_search_memory_tool_definition(), build_read_memory_source_tool_definition(),
                       build_bash_tool_definition()):
        registry.register(definition)
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(path),
        conversation_store=conversation, outbox=SQLiteOutboundStore(path), tools=registry,
        inference=InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'fixture', 'offline')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    runtime.core.system_one = SimpleNamespace(review_freeform_effect_claim=lambda **_: True)
    candidate = runtime.processor.owner_read_candidate
    assert candidate._owner_host_bash_enabled is False
    if trusted_local:
        from alphonse.agent_v2.daemon import _configure_trusted_local_bash
        _configure_trusted_local_bash(runtime, enabled=True)
    else:
        candidate.enable_owner_host_bash(allow_unprotected_for_testing=True)
    daemon = V2Daemon(runtime, daemon_id='fixture', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    socket_directory = None
    if through_ipc:
        # macOS Unix socket paths are short; keep this disposable path well
        # below the limit even when pytest's per-test directory is long.
        socket_directory = tempfile.TemporaryDirectory(prefix='al-owner-', dir='/tmp')
        socket_path = os.path.join(socket_directory.name, 'owner.sock')
        daemon.ipc = V2DaemonServer(daemon, socket_path=socket_path, experimental_owner_ingress=True)
    else:
        monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
        monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        if through_ipc:
            client = V2DaemonClient(socket_path)
            assert client.runtime_identity()['owner_host_bash_mode'] == (
                'trusted_local_unprotected' if trusted_local else 'experimental_unprotected')
            message_id = client.request('experimental_owner_ingress',
                prompt='Create a synthetic prospects note and check it.')['message_id']
        else:
            message_id = queue.enqueue_trusted_local_owner(prompt='Create a synthetic prospects note and check it.', peer_uid=os.getuid())
        deadline = time.monotonic() + 8
        status = 'pending'
        while time.monotonic() < deadline:
            status = (client.request('experimental_owner_status', message_id=message_id)['status']
                      if through_ipc else daemon.owner_message_status(message_id))
            if status not in {'pending', 'processing'}:
                break
            time.sleep(.01)
        assert status == 'ready'
        assert (users.ordinary_workspace(owner.user_id) / 'prospects.txt').read_text() == 'synthetic lead\n'
        from alphonse.agent_v2.owner_followups import OwnerFollowups
        held = OwnerFollowups(queue, owner.user_id, os.getuid())
        followup_id = 'd' * 32
        assert held.add(message_id, 'What does the note contain?', followup_id)['status'] == 'queued'
        draft = (client.request('experimental_owner_take_draft', message_id=message_id)
                 if through_ipc else daemon.take_owner_desktop_draft(message_id))
        assert draft['message'] == 'The note contains synthetic lead.'
        if through_ipc:
            client.request('experimental_owner_ack_draft', message_id=message_id,
                           acknowledgement_token=draft['acknowledgement_token'])
            assert client.request('experimental_owner_status', message_id=message_id)['status'] == 'client_acknowledged'
        else:
            daemon.acknowledge_owner_desktop_draft(message_id, draft['acknowledgement_token'])
        from alphonse.agent_v2.core.intelligence.simple_loop.owner_clarification import bind_owner_followup
        with queue._connect() as conn:
            assert bind_owner_followup(conn, owner.user_id, message_id)['message_id'] == message_id
            conn.execute('BEGIN')
            raw = json.loads(conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (message_id,)).fetchone()[0])
            for alteration in ('verified', 'missing', 'empty'):
                changed = json.loads(json.dumps(raw))
                if alteration == 'verified':
                    changed['metadata']['owner_host_bash_history'][0]['effect_status'] = 'verified'
                elif alteration == 'missing':
                    del changed['metadata']['owner_host_bash_history']
                else:
                    changed['metadata']['owner_host_bash_history'] = []
                conn.execute('UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id=?', (json.dumps(changed), message_id))
                with pytest.raises(ValueError, match='owner_question_context_unavailable'):
                    bind_owner_followup(conn, owner.user_id, message_id)
            conn.rollback()
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            status = daemon.owner_message_status(followup_id)
            if status not in {'unavailable', 'pending', 'processing'}:
                break
            time.sleep(.01)
        assert status == 'ready'
        followup = daemon.take_owner_desktop_draft(followup_id)
        daemon.acknowledge_owner_desktop_draft(followup_id, followup['acknowledgement_token'])
    finally:
        daemon.stop()
        if socket_directory is not None:
            socket_directory.cleanup()


def test_historical_host_output_loses_access_after_skill_revocation_or_change(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_atomic_checkpoint import _decision_digest, observation_digest
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_host_bash import valid_host_binding, authorized_host_skill_context
    monkeypatch.setattr('alphonse.agent_v2.core.intelligence.simple_loop.owner_host_bash.verify_owner_execution_claim',
                        lambda queue, receipt, task: queue.verify_trusted_local_owner_claim(receipt, task))
    from alphonse.agent_v2.core.messages.claim_fence import ClaimReceipt
    from alphonse.agent_v2.core.messages.experimental_claim_checkpoint import _claim_digest
    from alphonse.agent_v2.skill_resources import package_revision
    from alphonse.agent_v2.skills import SkillRecord

    directory = tmp_path / 'skill'
    directory.mkdir()
    (directory / 'SKILL.md').write_text('synthetic instructions', encoding='utf-8')
    skill = SkillRecord('skill:demo', 'demo', 'Synthetic', str(directory), 'synthetic instructions')
    class Skills:
        allowed = True
        def can_use(self, skill_id, actor):
            return self.allowed and skill_id == skill.skill_id and actor == 'owner'
        def get(self, skill_id):
            return skill if skill_id == skill.skill_id else None
    skills = Skills()
    task = SimpleNamespace(user='owner', metadata={'v3_skill_selection': {
        'selected_skill_ids': [skill.skill_id],
        'packages': {skill.skill_id: {'root': str(directory), 'revision': package_revision(skill)}}}})
    queue = SimpleNamespace(verify_trusted_local_owner_claim=lambda *_: True)
    receipt = ClaimReceipt('message', 'owner', 'worker', 'a' * 32)
    checkpoint = SimpleNamespace(task=task, queue=queue, receipt=receipt)
    context = SimpleNamespace(skill_store=skills, artifact_resources=None,
        cancellation_checker=lambda: False, _owner_host_bash_enabled=True)
    args = {'command': 'cat synthetic.txt', 'skill_id': skill.skill_id}
    output = {'exit_code': 0, 'timed_out': False, 'cancelled': False,
              'effect_status': 'unverified', 'stdout': 'synthetic data', 'stderr': ''}
    digest = observation_digest(output)
    binding = {'actor': 'owner', 'tool_id': 'native.bash', 'arguments': args,
               'observation_sha256': digest, 'host_bash_observation': output}
    history = [{'claim_digest': _claim_digest(receipt), 'arguments_sha256': _decision_digest(args),
                'observation_sha256': digest, 'effect_status': 'unverified'}]
    assert valid_host_binding(checkpoint, context, binding, history)
    skills.allowed = False
    assert not valid_host_binding(checkpoint, context, binding, history)
    skills.allowed = True
    (directory / 'resource.txt').write_text('changed', encoding='utf-8')
    assert not valid_host_binding(checkpoint, context, binding, history)
    assert not authorized_host_skill_context(context, task, {**args, 'binding_names': ['account']})
    assert authorized_host_skill_context(context, task, {'command': 'printf hello'})
