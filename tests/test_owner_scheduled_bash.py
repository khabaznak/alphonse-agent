"""Scheduled Bash keeps its own owner provenance and the trusted-local opt-in."""
from datetime import datetime, timezone
import json
import shlex
import time

import pytest

from test_owner_schedule_origin import fixture as desktop_schedule
from test_owner_telegram_handoff import ready
from test_owner_telegram_schedule_origin import telegram_schedule
from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.intelligence.simple_loop.owner_schedule_origin import stage_due_occurrence_locked
from alphonse.agent_v2.core.intelligence.simple_loop.owner_atomic_checkpoint import OwnerAtomicCheckpoint
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_checkpoint import CheckpointConflict, ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.simple_loop.owner_host_bash import dispatch_owner_host_bash
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages.queue import MessageSelector
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native.memory import build_read_memory_source_tool_definition, build_search_memory_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore
from test_owner_atomic_checkpoint import make_budget, reserve_selected_tool


@pytest.mark.parametrize('origin,enabled', [('desktop', True), ('telegram', True), ('desktop', False), ('telegram', False)])
def test_scheduled_bash_is_owner_scoped_and_opt_in(request, tmp_path, monkeypatch, origin, enabled):
    queue, schedule, due = request.getfixturevalue('desktop_schedule' if origin == 'desktop' else 'telegram_schedule')
    queue._owner_schedule_clock = lambda: due
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        message_id = stage_due_occurrence_locked(conn, schedule.scheduled_task_id, now=due)
        # The Desktop schedule fixture does not run its original request to
        # completion; remove only that synthetic request from this disposable DB.
        if origin == 'desktop':
            conn.execute('DELETE FROM v2_inbound_messages WHERE message_id!=?', (message_id,))
    source = tmp_path / 'pantry.txt'
    source.write_text('Synthetic pantry has 73 cans.\n')

    class Provider:
        supports_image_input = False
        calls = 0
        def generate_json(self, req):
            self.calls += 1
            assert ('native.bash' in req.prompt) is enabled
            decision = ({'kind': 'tool', 'tool_id': 'native.bash',
                         'arguments': {'command': 'cat ' + shlex.quote(str(source))}, 'text': None}
                        if enabled and self.calls == 1 else
                        {'kind': 'answer', 'tool_id': None, 'arguments': None,
                         'text': 'The synthetic pantry has 73 cans.' if enabled else 'Bash is not available.'})
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    class Listener:
        sent = []
        def send_reply(self, **kwargs): self.sent.append(kwargs); return 'synthetic-reply'
        def stop(self, *, timeout=12): return True

    tools = InMemoryToolRegistry()
    tools.register(build_search_memory_tool_definition())
    tools.register(build_read_memory_source_tool_definition())
    tools.register(build_bash_tool_definition())
    users = V2UserStore(queue.db_path)
    users.set_workspace_root(tmp_path / 'workspace')
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(queue.db_path),
        schedule_store=ScheduledTaskStore(queue.db_path), conversation_store=SQLiteConversationStore(queue.db_path),
        outbox=SQLiteOutboundStore(queue.db_path), tools=tools,
        inference=InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'fixture', 'offline')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    if enabled:
        runtime.processor.owner_read_candidate.enable_owner_host_bash(trusted_local_development=True)
    daemon = V2Daemon(runtime, daemon_id='scheduled-bash-test', experimental_owner_only=True)
    listener = Listener()
    if origin == 'telegram': daemon._owner_telegram_listener = listener
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            with queue._connect() as conn:
                row = conn.execute('SELECT state FROM v2_owner_schedule_occurrences WHERE message_id=?', (message_id,)).fetchone()
            if row and row['state'] != 'queued': break
            time.sleep(.02)
        assert row and row['state'] == 'completed'
        with queue._connect() as conn:
            saved = json.loads(conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (message_id,)).fetchone()[0])
            observations = saved['metadata']['owner_read_observations']
            bash = [o for o in observations.values() if o.get('tool_id') == 'native.bash']
            assert len(bash) == (1 if enabled else 0)
            if enabled:
                assert '73 cans' in bash[0]['host_bash_observation']['stdout']
            result = conn.execute('SELECT content FROM v2_conversation_events WHERE source_message_id=?',
                                  ('outbound:scheduled:' + message_id,)).fetchone()
            assert result is not None
        assert source.read_text() == 'Synthetic pantry has 73 cans.\n'
        if origin == 'telegram': assert len(listener.sent) == 1
    finally:
        daemon.stop()


@pytest.mark.parametrize('origin,revoked', [('desktop', False), ('telegram', False), ('desktop', True), ('telegram', True)])
def test_scheduled_bash_does_not_replay_or_run_after_revocation(request, tmp_path, origin, revoked):
    queue, schedule, due = request.getfixturevalue('desktop_schedule' if origin == 'desktop' else 'telegram_schedule')
    queue._owner_schedule_clock = lambda: due
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        message_id = stage_due_occurrence_locked(conn, schedule.scheduled_task_id, now=due)
    acquired = queue.claim_next(selector=MessageSelector(message_id=message_id), lease_owner='scheduled-bash', lease_seconds=180)
    task = TaskState.from_message(acquired.message, message_id=message_id)
    task.task_id = message_id
    atomic = OwnerAtomicCheckpoint(queue, ExperimentalCheckpointStore(SQLiteQuestionStore(queue.db_path)),
                                   task, acquired.claim_receipt, clock=lambda: datetime.now(timezone.utc))
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    target = tmp_path / 'synthetic-effect.txt'
    args = {'command': 'printf x >> ' + shlex.quote(str(target)), 'cwd': str(tmp_path)}
    reserve_selected_tool(atomic, start, 'native.bash', args)
    context = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt, cancellation_checker=lambda: False)
    if revoked:
        with queue._connect() as conn:
            if origin == 'telegram': conn.execute('UPDATE v2_user_addresses SET is_active=0')
            else: conn.execute("UPDATE v2_scheduled_tasks SET status='cancelled' WHERE scheduled_task_id=?", (schedule.scheduled_task_id,))
        with pytest.raises(PermissionError, match='owner_host_bash_unavailable'):
            dispatch_owner_host_bash(atomic=atomic, context=context, arguments=args, reservation_id='read-1', authorize=lambda *_: True)
        assert not target.exists()
    else:
        result = dispatch_owner_host_bash(atomic=atomic, context=context, arguments=args, reservation_id='read-1', authorize=lambda *_: True)
        assert result['exit_code'] == 0 and target.read_text() == 'x'
        with pytest.raises(CheckpointConflict, match='owner_atomic_bash_dispatch_unavailable'):
            dispatch_owner_host_bash(atomic=atomic, context=context, arguments=args, reservation_id='read-1', authorize=lambda *_: True)
        assert target.read_text() == 'x'
