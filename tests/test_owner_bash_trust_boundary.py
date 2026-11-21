"""Disposable demonstrations of host Bash's authority boundary.

These tests do not grant shell access or modify a household installation.
"""
import json
import os
from pathlib import Path
import shlex
import socket
import sqlite3
import sys
import tempfile
import threading
from types import SimpleNamespace

from alphonse.agent_v2.core.tools.registry.native.bash import execute_bash
from alphonse.agent_v2.ipc import V2DaemonServer
from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages.claim_fence import ClaimReceipt
from alphonse.agent_v2.core.messages.queue import MessageSelector
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.users import V2UserStore


def test_host_bash_child_passes_same_local_owner_peer_check():
    """Same OS UID is not evidence that the human initiated the IPC call."""
    with tempfile.TemporaryDirectory(prefix='owner-bash-peer-', dir='/tmp') as directory:
        path = str(Path(directory) / 's.sock')
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(path)
            os.chmod(path, 0o600)
            listener.listen(1)
            listener.settimeout(10)
            script = 'import socket,sys; s=socket.socket(socket.AF_UNIX); s.connect(sys.argv[1]); s.sendall(b"x"); s.close()'
            command = f'{shlex.quote(sys.executable)} -c {shlex.quote(script)} {shlex.quote(path)}'
            result = {}
            def execute():
                try:
                    result['value'] = execute_bash({'command': command, 'timeout_seconds': 5})
                except BaseException as exc:
                    result['error'] = exc
            thread = threading.Thread(target=execute)
            thread.start()
            try:
                connection, _ = listener.accept()
                with connection:
                    assert V2DaemonServer._local_peer_uid(connection) == os.getuid()
                    assert connection.recv(1) == b'x'
            finally:
                thread.join(timeout=10)
            assert not thread.is_alive()
            assert 'error' not in result
            assert result['value']['exit_code'] == 0


def test_host_bash_can_modify_same_uid_capability_database():
    """A grant stored under the shell's writable OS identity is not a fence."""
    with tempfile.TemporaryDirectory(prefix='owner-bash-db-', dir='/tmp') as directory:
        path = Path(directory) / 'fixture.sqlite'
        with sqlite3.connect(path) as conn:
            conn.execute('CREATE TABLE capability (enabled INTEGER NOT NULL)')
            conn.execute('INSERT INTO capability VALUES (0)')
        path.chmod(0o600)
        script = 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); c.execute("UPDATE capability SET enabled=1"); c.commit()'
        command = f'{shlex.quote(sys.executable)} -c {shlex.quote(script)} {shlex.quote(str(path))}'
        result = execute_bash({'command': command, 'timeout_seconds': 5})
        assert result['exit_code'] == 0
        with sqlite3.connect(path) as conn:
            assert conn.execute('SELECT enabled FROM capability').fetchone()[0] == 1


def test_host_bash_can_submit_a_real_owner_ingress_without_a_desktop():
    """Exercise the actual endpoint, not just the kernel peer check.

    This is an observed current limitation, not a desired authority policy.
    The listener and every record are disposable; the owner worker is not run.
    """
    with tempfile.TemporaryDirectory(prefix='owner-ingress-boundary-', dir='/tmp') as directory:
        root = Path(directory)
        db = root / 'fixture.sqlite'
        users = V2UserStore(db)
        owner = users.onboard(display_name='Synthetic owner', users_root=root / 'users')
        SQLiteConversationStore(db)
        queue = SQLiteMessageQueue(db, experimental_claim_fencing=True)
        daemon = SimpleNamespace(experimental_owner_only=True, owner_processing_ready=lambda: True,
                                 runtime=SimpleNamespace(queue=queue))
        server = V2DaemonServer(daemon, root / 'owner.sock', experimental_owner_ingress=True)
        server.start()
        try:
            script = '''import json,socket,sys
p={'version':1,'request_id':'synthetic','method':'experimental_owner_ingress',
   'params':{'prompt':'Synthetic request from Bash, not a Desktop interaction'}}
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.connect(sys.argv[1])
s.sendall((json.dumps(p)+'\\n').encode()); data=b''
while b'\\n' not in data:
 chunk=s.recv(65536)
 if not chunk: break
 data+=chunk
print(data.decode()); s.close()
'''
            command = f'{shlex.quote(sys.executable)} -c {shlex.quote(script)} {shlex.quote(str(root / "owner.sock"))}'
            result = execute_bash({'command': command, 'timeout_seconds': 5})
            assert result['exit_code'] == 0
            response = json.loads(result['stdout'])
            assert response['ok'] is True
            message_id = response['result']['message_id']
            with queue._connect() as conn:
                row = conn.execute('''SELECT q.prompt,q.user_id,o.owner_id FROM v2_inbound_messages q
                    JOIN v2_trusted_owner_ingress o ON o.message_id=q.message_id WHERE q.message_id=?''',
                    (message_id,)).fetchone()
            assert row is not None and row['user_id'] == row['owner_id'] == owner.user_id
            assert row['prompt'] == 'Synthetic request from Bash, not a Desktop interaction'
        finally:
            server.stop()


def test_host_bash_can_replace_a_live_owner_claim_in_same_uid_database():
    """Prove a forged claim passes the real claim fence after a direct DB edit."""
    with tempfile.TemporaryDirectory(prefix='owner-claim-boundary-', dir='/tmp') as directory:
        root = Path(directory)
        db = root / 'fixture.sqlite'
        users = V2UserStore(db)
        owner = users.onboard(display_name='Synthetic owner', users_root=root / 'users')
        SQLiteConversationStore(db)
        queue = SQLiteMessageQueue(db, experimental_claim_fencing=True)
        message_id = queue.enqueue_trusted_local_owner(prompt='Synthetic original request', peer_uid=os.getuid())
        acquired = queue.claim_next(selector=MessageSelector(message_id=message_id),
                                    lease_owner='legitimate-worker', lease_seconds=120)
        assert acquired is not None
        task = TaskState.from_message(acquired.message, message_id=message_id)
        task.task_id = message_id
        assert queue.verify_trusted_local_owner_claim(acquired.claim_receipt, task)
        script = '''import sqlite3,sys
c=sqlite3.connect(sys.argv[1]); c.execute("UPDATE v2_inbound_messages SET lease_owner=?,claim_generation=? WHERE message_id=?", ("forged-worker","a"*32,sys.argv[2])); c.commit(); c.close()
'''
        command = f'{shlex.quote(sys.executable)} -c {shlex.quote(script)} {shlex.quote(str(db))} {shlex.quote(message_id)}'
        result = execute_bash({'command': command, 'timeout_seconds': 5})
        assert result['exit_code'] == 0
        forged = ClaimReceipt(message_id=message_id, user_id=owner.user_id,
                              lease_owner='forged-worker', claim_generation='a'*32)
        assert not queue.verify_trusted_local_owner_claim(acquired.claim_receipt, task)
        assert queue.verify_trusted_local_owner_claim(forged, task)
