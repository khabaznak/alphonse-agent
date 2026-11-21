"""Scripted owner-worker claim-loss exercise in a disposable Bash container.

This is a wiring/recovery diagnostic, not a live-model or normal-host test.
The fixture expires a synthetic queue claim after a synthetic command starts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from unittest.mock import patch

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_checkpoint import STATE_KEY
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native.memory import build_read_memory_source_tool_definition, build_search_memory_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.evaluation.dojo_bash import DojoBash
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore

COMMAND = 'printf started > /workspace/started; sleep 20; printf late > /workspace/late'


class ScriptedProvider:
    supports_image_input = False

    def __init__(self):
        self.calls = 0

    def generate_json(self, _request):
        self.calls += 1
        decision = {'kind': 'tool', 'tool_id': 'native.bash',
                    'arguments': {'command': COMMAND}, 'text': None}
        return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                               usage={'input_tokens': 100, 'output_tokens': 20})


def grade_attempt(*, marker_seen, updated, events, cleanup_verified, status, provider_calls, inflight, pending,
                  recovery_items, message_id):
    """Require observed cancellation and retained exact in-flight action."""
    checks = {'command_start_observed': marker_seen is True,
              'claim_expired_after_start': updated == 1,
              'shell_cancelled': bool(events and events[-1].get('status') == 'cancelled'),
              'container_cleanup_verified': cleanup_verified is True,
              'no_ready_draft': status == 'failed',
              'single_model_call': provider_calls == 1,
              'owner_sees_unconfirmed_action': isinstance(recovery_items, list) and any(isinstance(item, dict) and item.get('message_id') == message_id
                  and item.get('status') == 'action_unconfirmed' and item.get('can_abandon') is False
                  for item in recovery_items),
              'unresolved_action_preserved': bool(inflight and isinstance(pending, dict)
                                                  and pending.get('reservation_id') == inflight
                                                  and pending.get('tool_id') == 'native.bash')}
    return {'passed': all(checks.values()), 'checks': checks}


def run(*, image_id: str, docker_socket: str, emit):
    """Observe claim loss after dispatch; preserve effects as uncertain."""
    started = time.monotonic()
    provider = ScriptedProvider()
    events = []
    with tempfile.TemporaryDirectory(prefix='owner-dojo-interrupt-', dir='/tmp') as directory:
        root = Path(directory)
        env = {k: v for k, v in os.environ.items() if not k.startswith('ALPHONSE_')}
        env['ALPHONSE_V2_DB_PATH'] = str(root / 'fixture.sqlite')
        env['ALPHONSE_V2_SOCKET_PATH'] = str(root / 'owner.sock')
        with patch.dict(os.environ, env, clear=True), patch.object(Path, 'home', classmethod(lambda cls: root)):
            path = root / 'fixture.sqlite'
            users = V2UserStore(path)
            owner = users.onboard(display_name='Synthetic owner', users_root=root / 'users')
            queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
            conversation = SQLiteConversationStore(path)
            tools = InMemoryToolRegistry()
            tools.register(build_search_memory_tool_definition())
            tools.register(build_read_memory_source_tool_definition())
            tools.register(build_bash_tool_definition())
            router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'scripted', 'dojo-interruption'))
            runtime = build_runtime_host(user_store=users, messages=queue,
                question_store=SQLiteQuestionStore(path), conversation_store=conversation,
                outbox=SQLiteOutboundStore(path), tools=tools, inference=router,
                coding_skill_service=object(), experimental_owner_read=True,
                owner_selected_draft_sink=lambda *_: None)
            with DojoBash(image_id=image_id, docker_socket=docker_socket, timeout_seconds=15) as shell:
                original_execute = shell.execute
                dispatched = threading.Event()
                returned = threading.Event()
                def observe(command, **kwargs):
                    emit({'event': 'bash_started'})
                    dispatched.set()
                    try:
                        result = original_execute(command, **kwargs)
                        events.append(result)
                        emit({'event': 'bash_finished', **result})
                        return result
                    except Exception as exc:
                        events.append({'status': 'error', 'error_type': type(exc).__name__})
                        emit({'event': 'bash_error', 'error_type': type(exc).__name__})
                        raise
                    finally:
                        returned.set()
                shell.execute = observe
                runtime.processor.owner_read_candidate.enable_dojo_bash(shell)
                daemon = V2Daemon(runtime, daemon_id='dojo-interruption', experimental_owner_only=True)
                daemon.ipc = V2DaemonServer(daemon, socket_path=root / 'owner.sock')
                try:
                    daemon.start()
                    client = V2DaemonClient(root / 'owner.sock')
                    message_id = client.request('experimental_owner_ingress', prompt='Run the synthetic interruption exercise.')['message_id']
                    if not dispatched.wait(timeout=10):
                        raise RuntimeError('dojo_shell_not_dispatched')
                    deadline = time.monotonic() + 8
                    marker_seen = False
                    while time.monotonic() < deadline:
                        check = original_execute('test -f /workspace/started')
                        if check['status'] == 'completed':
                            marker_seen = True
                            break
                        time.sleep(.05)
                    if not marker_seen:
                        raise RuntimeError('dojo_command_start_not_observed')
                    # Only this disposable database is changed; expiry simulates
                    # lost ownership during the running shell command.
                    with queue._connect() as conn:
                        updated = conn.execute("UPDATE v2_inbound_messages SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE message_id=? AND status='processing'", (message_id,)).rowcount
                    if updated != 1:
                        raise RuntimeError('dojo_claim_expiry_failed')
                    emit({'event': 'claim_expired_after_observed_dispatch'})
                    deadline = time.monotonic() + 10
                    status = 'pending'
                    while time.monotonic() < deadline:
                        status = client.request('experimental_owner_status', message_id=message_id)['status']
                        if status not in {'pending', 'processing'}:
                            break
                        time.sleep(.05)
                    returned.wait(timeout=10)
                    recovery = client.request('experimental_owner_recovery_list').get('items', [])
                    with queue._connect() as conn:
                        row = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (message_id,)).fetchone()
                    raw = json.loads(row['task_state_json']) if row else {}
                    envelope = raw.get('metadata', {}).get(STATE_KEY, {})
                    pending = raw.get('metadata', {}).get('owner_read_pending_action', {})
                    inflight = envelope.get('state', {}).get('inflight')
                    graded = grade_attempt(marker_seen=marker_seen, updated=updated, events=events,
                        cleanup_verified=shell.cleanup_verified, status=status, provider_calls=provider.calls,
                        inflight=inflight, pending=pending, recovery_items=recovery, message_id=message_id)
                    return {'case': 'owner-dojo-claim-loss', 'status': status,
                            **graded,
                            'provider_calls': provider.calls, 'shell_results': events,
                            'elapsed_seconds': round(time.monotonic() - started, 3),
                            'limitations': ['Scripted inference; this does not establish model behavior.',
                                'Uses injected disposable container Bash, not household host Bash.',
                                'Checks preserved in-flight state, not cross-restart reconciliation.',
                                'Container cleanup cannot undo an external effect in a real host environment.']}
                finally:
                    daemon.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--docker-socket', required=True)
    args = parser.parse_args()
    with args.journal.open('x', encoding='utf-8') as stream:
        def emit(row):
            stream.write(json.dumps(row, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        emit({'event': 'started', 'case': 'owner-dojo-claim-loss', 'inference': 'scripted',
              'scope': 'synthetic owner IPC, disposable database and isolated container'})
        fd = os.open(args.journal.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            result = run(image_id=args.image_id, docker_socket=args.docker_socket, emit=emit)
        except Exception as exc:
            emit({'event': 'finished', 'outcome': 'infrastructure_error', 'passed': None,
                  'error_type': type(exc).__name__})
            raise
        emit({'event': 'finished', **result})
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
