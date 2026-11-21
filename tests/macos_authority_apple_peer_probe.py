"""Opt-in macOS audit-token probe with an existing Apple system binary.

Compiles into a temporary directory, starts two disposable Unix socket
listeners, and runs /usr/bin/curl as the client. Sends only a fixed HTTP reply.
It checks verification mechanics, not authentication of Alphonse clients.
"""
from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    if os.uname().sysname != 'Darwin':
        raise SystemExit('macOS required')
    with tempfile.TemporaryDirectory(prefix='alphonse-apple-peer-', dir='/private/tmp') as directory:
        root = Path(directory)
        executable = root / 'verify'
        subprocess.run(['xcrun', 'swiftc', '-parse-as-library', '-module-cache-path', str(root / 'cache'),
                        str(ROOT / 'alphonse/agent_v2/authority_service/macos/PeerPolicy.swift'),
                        str(ROOT / 'alphonse/agent_v2/authority_service/macos/LocalPeerVerifier.swift'),
                        str(ROOT / 'tests/macos_authority_apple_peer_verifier.swift'),
                        '-o', str(executable)], check=True, timeout=120)
        for index, mode in enumerate(('correct', 'wrong')):
            path = root / ('s%d' % index)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(path))
                server.listen(1)
                server.settimeout(10)
                client = subprocess.Popen(['/usr/bin/curl', '--silent', '--show-error', '--max-time', '10',
                                           '--unix-socket', str(path), 'http://localhost/'],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    connection, _ = server.accept()
                    with connection:
                        verification = subprocess.run([str(executable), str(connection.fileno()), mode],
                            pass_fds=(connection.fileno(),), capture_output=True, text=True, timeout=10)
                        if verification.returncode != 0:
                            raise RuntimeError('apple_peer_verification_failed:' + mode + ':'
                                               + verification.stderr[-500:])
                        # No request bytes are read and no owner data is sent.
                        connection.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n')
                    stdout, stderr = client.communicate(timeout=10)
                    if client.returncode:
                        raise RuntimeError('apple_system_client_failed:' + str(client.returncode)
                                           + ':' + stderr.decode(errors='replace')[-200:])
                    print(mode + ': ' + verification.stdout.strip(), flush=True)
                finally:
                    if client.poll() is None:
                        client.kill()
                        client.communicate(timeout=5)
    print('temporary sockets and verifier removed', flush=True)


if __name__ == '__main__':
    main()
