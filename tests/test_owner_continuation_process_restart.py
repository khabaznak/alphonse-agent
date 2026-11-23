"""Disposable true-process restart test for trusted-local continuation."""
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.ipc import V2DaemonClient
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def _daemon_process(db_path, socket_path, work_dir, ready):
    """Rebuild the worker independently, without accessing the household database."""
    class Provider:
        supports_image_input = False
        def generate_json(self, request):
            count_path = Path(work_dir) / 'model-count.txt'
            count = int(count_path.read_text()) if count_path.exists() else 0
            count += 1
            count_path.write_text(str(count))
            if count in (1, 2):
                command = 'printf x >> result.txt' if count == 1 else 'cat result.txt'
                decision = {'kind': 'tool', 'tool_id': 'native.bash',
                            'arguments': {'command': command, 'cwd': work_dir}, 'text': None}
            else:
                decision = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'The result is x.'}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    users = V2UserStore(db_path)
    queue = SQLiteMessageQueue(db_path, experimental_claim_fencing=True)
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(db_path),
        conversation_store=SQLiteConversationStore(db_path), outbox=SQLiteOutboundStore(db_path),
        inference=InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'fixture', 'offline')),
        experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    runtime.processor.owner_read_candidate.enable_owner_host_bash(trusted_local_development=True)
    daemon = V2Daemon(runtime, daemon_id=f'fixture-{os.getpid()}', experimental_owner_only=True)
    daemon._continuation_experimental_enabled = True
    daemon._acquire_single_instance_lock = lambda: None
    daemon._release_single_instance_lock = lambda: None
    daemon.ipc.socket_path = Path(socket_path)
    daemon.start()
    ready.send(True)
    try:
        while True:
            time.sleep(1)
    finally:
        daemon.stop()


def _wait_until(callback, process, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process.is_alive():
            raise AssertionError(f'disposable daemon exited: {process.exitcode}')
        try:
            result = callback()
            if result:
                return result
        except (OSError, RuntimeError, KeyError):
            pass
        time.sleep(.03)
    raise AssertionError('disposable daemon did not reach expected state')


def test_true_process_restart_retains_paused_bash_work(tmp_path):
    db = str(tmp_path / 'owner.db')
    users = V2UserStore(db)
    users.set_workspace_root(tmp_path / 'workspace')
    users.onboard(display_name='Disposable owner', users_root=tmp_path / 'users')
    users._set_setting('owner_continuation_checkpoint', json.dumps({'metric':'actions','interval':1}))
    context = multiprocessing.get_context('spawn')
    with tempfile.TemporaryDirectory(prefix='alps-', dir='/tmp') as socket_dir:
        socket_path = str(Path(socket_dir) / 's.sock')
        processes = []
        def start():
            parent, child = context.Pipe(duplex=False)
            process = context.Process(target=_daemon_process, args=(db, socket_path, str(tmp_path), child))
            process.start()
            child.close()
            processes.append(process)
            assert parent.poll(15) and parent.recv() is True
            parent.close()
            return process
        try:
            first = start()
            client = V2DaemonClient(socket_path)
            message = client.request('experimental_owner_ingress', prompt='Read the disposable record.')['message_id']
            _wait_until(lambda: client.request('experimental_owner_status', message_id=message).get('status') == 'continuation_paused', first)
            assert (tmp_path / 'result.txt').read_text() == 'x'
            first.kill()
            first.join(5)
            assert not first.is_alive()
            second = start()
            assert second.pid != first.pid
            client = V2DaemonClient(socket_path)
            for _ in range(2):
                items = _wait_until(lambda: client.request('experimental_owner_continuation_list')['items'], second)
                item = next(item for item in items if item['message_id'] == message)
                assert client.request('experimental_owner_continuation_choose', message_id=message,
                    checkpoint_version=item['checkpoint_version'], approve=True)['status'] == 'accepted'
            _wait_until(lambda: client.request('experimental_owner_status', message_id=message).get('status') == 'ready', second)
            draft = client.request('experimental_owner_take_draft', message_id=message)
            assert 'The result is x.' in draft['message']
            assert (tmp_path / 'result.txt').read_text() == 'x'
            assert (tmp_path / 'model-count.txt').read_text() == '3'
        finally:
            for process in processes:
                if process.is_alive():
                    process.kill()
                process.join(5)
