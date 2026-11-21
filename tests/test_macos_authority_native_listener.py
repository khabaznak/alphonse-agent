"""Exercise real Darwin listener inspection; not a production identity test."""
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile

import pytest

from alphonse.agent_v2.authority_service.macos.peer_verifier import MacOSSocketPeerVerifier


@pytest.mark.skipif(sys.platform != 'darwin' or not shutil.which('xcrun'), reason='requires macOS Swift')
def test_native_listener_check_does_not_accept_or_change_backlog(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'alphonse/agent_v2/authority_service/macos'
    library = tmp_path / 'verifier.dylib'
    subprocess.run(['xcrun', 'swiftc', '-emit-library', '-module-cache-path', str(tmp_path / 'cache'),
                    *(str(source / name) for name in ('PeerPolicy.swift', 'LocalPeerVerifier.swift', 'PeerVerifierBridge.swift')),
                    '-o', str(library)], check=True, timeout=90, capture_output=True)
    verifier = MacOSSocketPeerVerifier(str(library), desktop_team='ABCDEFGHI1', desktop_identifier='test.desktop',
                                       worker_team='ABCDEFGHI1', worker_identifier='test.worker')
    with tempfile.TemporaryDirectory(prefix='alpha-listen-', dir='/private/tmp') as directory:
        path = str(Path(directory) / 's')
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(path)
            assert verifier.is_listening(listener) is False
            listener.listen(4)
            assert verifier.is_listening(listener) is True
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(path)
                assert verifier.is_listening(listener) is True
                # The pending connection remains available after the check.
                accepted, _ = listener.accept()
                accepted.close()
        assert verifier.is_listening(listener) is False
