import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

import pytest

from alphonse.agent_v2.authority_service.macos import local_transport


pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason='macOS getpeereid transport')


@pytest.fixture
def socket_root():
    # Darwin's sockaddr_un path limit is much shorter than pytest's tmp path.
    with tempfile.TemporaryDirectory(prefix='alpha-peer-', dir='/private/tmp') as directory:
        yield Path(directory)


def test_same_user_service_identity_is_rejected_before_connect():
    with pytest.raises(local_transport.AuthorityConnectionError, match='authority_service_identity_invalid'):
        local_transport.connect_verified_service('/tmp/nonexistent-authority.sock', service_uid=os.geteuid())
    with pytest.raises(local_transport.AuthorityConnectionError, match='authority_service_identity_invalid'):
        local_transport.connect_verified_service('/tmp/nonexistent-authority.sock', service_uid=-1)


def test_kernel_peer_identity_from_connected_unix_socket():
    left, right = socket.socketpair()
    try:
        assert local_transport._peer_effective_uid(left) == os.geteuid()
        assert local_transport._peer_effective_uid(right) == os.geteuid()
    finally:
        left.close()
        right.close()


def test_separate_same_user_impostor_receives_no_first_message(socket_root):
    path = socket_root / 'peer.sock'
    script = '''import socket,sys
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.bind(sys.argv[1]); s.listen(1)
print("ready", flush=True)
c,_=s.accept(); c.settimeout(3)
data=c.recv(1024)
print("received:"+str(len(data)), flush=True)
c.close(); s.close()
'''
    process = subprocess.Popen([sys.executable, '-c', script, str(path)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == 'ready'
        # The synthetic expected UID differs from this user's actual UID. The
        # test does not require, pretend, or provision a genuine service user.
        expected = os.geteuid() + 1
        with pytest.raises(local_transport.AuthorityConnectionError, match='authority_service_identity_mismatch'):
            local_transport.connect_verified_service(path, service_uid=expected)
        output, error = process.communicate(timeout=5)
        assert process.returncode == 0, error
        assert output.strip() == 'received:0'
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def test_verified_connection_returns_the_same_checked_socket(socket_root, monkeypatch):
    # This only tests call ordering. The trusted UID is simulated; real
    # cross-identity acceptance requires an independently provisioned OS user.
    path = socket_root / 'service.sock'
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)
    expected = os.geteuid() + 1
    monkeypatch.setattr(local_transport, '_peer_effective_uid', lambda _: expected)
    connection = local_transport.connect_verified_service(path, service_uid=expected)
    peer, _ = listener.accept()
    try:
        connection.sendall(b'synthetic request')
        assert peer.recv(1024) == b'synthetic request'
    finally:
        connection.close()
        peer.close()
        listener.close()
