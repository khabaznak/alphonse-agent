"""Opt-in, disposable macOS pinned-CDHash and hardened-runtime probe.

Compiles and ad-hoc signs synthetic binaries only; never changes the user's
keychains, trust settings or launchd. Passing this test does not authenticate a
person, secure writable app resources, or establish a production boundary.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphonse.agent_v2.authority_service.macos.peer_verifier import MacOSSocketPeerVerifier

CLIENT = r'''#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <string.h>
int main(int argc, char **argv) {
 if (argc != 2) return 2;
 int fd = socket(AF_UNIX, SOCK_STREAM, 0);
 struct sockaddr_un address = {0}; address.sun_family = AF_UNIX;
 strncpy(address.sun_path, argv[1], sizeof(address.sun_path)-1);
 if (connect(fd, (struct sockaddr *)&address, sizeof(address))) return 3;
 char c; read(fd, &c, 1); close(fd); return RETURN_VALUE;
}'''
INJECT = r'''#include <stdio.h>
#include <stdlib.h>
__attribute__((constructor)) static void loaded(void) {
 const char *p=getenv("PROBE_MARKER"); if (p) { FILE *f=fopen(p,"w"); if(f){fputs("injected",f);fclose(f);} }
}'''


def command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=120)


def code_hash(path):
    result = command(['/usr/bin/codesign', '-dv', '--verbose=4', str(path)])
    match = re.search(r'^CDHash=([0-9a-f]{40})$', result.stderr, re.MULTILINE)
    if not match:
        raise RuntimeError('fixture_cdhash_unavailable')
    return match.group(1), result.stderr


def main():
    if sys.platform != 'darwin':
        raise SystemExit('macOS required')
    with tempfile.TemporaryDirectory(prefix='alphonse-cdhash-', dir='/private/tmp') as directory:
        root = Path(directory)
        paths = []
        for index in (0, 1):
            source = root / ('client%d.c' % index)
            source.write_text(CLIENT.replace('RETURN_VALUE', '0' if index == 0 else 'argc == 99'))
            binary = root / ('client%d' % index)
            command(['/usr/bin/clang', str(source), '-o', str(binary)])
            command(['/usr/bin/codesign', '--force', '--sign', '-', '--options', 'runtime',
                     '--identifier', 'org.alphonse.synthetic', str(binary)])
            digest, info = code_hash(binary)
            if 'adhoc,runtime' not in info:
                raise RuntimeError('fixture_hardened_signature_unavailable')
            paths.append((binary, digest))
        if paths[0][1] == paths[1][1]:
            raise RuntimeError('fixture_hashes_not_distinct')
        clone = root / 'identical-copy'
        shutil.copy2(paths[0][0], clone)
        dylib = root / 'peer-verifier.dylib'
        sources = ROOT / 'alphonse/agent_v2/authority_service/macos'
        command(['/usr/bin/xcrun', 'swiftc', '-emit-library', '-module-cache-path', str(root / 'cache'),
                 str(sources / 'PeerPolicy.swift'), str(sources / 'LocalPeerVerifier.swift'),
                 str(sources / 'PeerVerifierBridge.swift'), '-o', str(dylib)])
        verifier = MacOSSocketPeerVerifier(dylib, desktop_cdhash=paths[0][1], worker_cdhash=paths[1][1])
        injection_source = root / 'inject.c'
        injection_source.write_text(INJECT)
        injected_library = root / 'inject.dylib'
        command(['/usr/bin/clang', '-dynamiclib', str(injection_source), '-o', str(injected_library)])
        plain = root / 'unhardened'
        command(['/usr/bin/clang', str(root / 'client0.c'), '-o', str(plain)])
        command(['/usr/bin/codesign', '--force', '--sign', '-', '--identifier', 'org.alphonse.synthetic', str(plain)])
        plain_hash, _ = code_hash(plain)
        plain_verifier = MacOSSocketPeerVerifier(dylib, desktop_cdhash=plain_hash, worker_cdhash=paths[1][1])
        risky = root / 'risky-entitlements'
        command(['/usr/bin/clang', str(root / 'client0.c'), '-o', str(risky)])
        entitlements = root / 'entitlements.plist'
        entitlements.write_text('<?xml version="1.0" encoding="UTF-8"?>'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
            '<plist version="1.0"><dict><key>com.apple.security.get-task-allow</key><true/>'
            '<key>com.apple.security.cs.disable-library-validation</key><true/></dict></plist>')
        command(['/usr/bin/codesign', '--force', '--sign', '-', '--options', 'runtime',
                 '--entitlements', str(entitlements), '--identifier', 'org.alphonse.synthetic', str(risky)])
        risky_hash, _ = code_hash(risky)
        risky_verifier = MacOSSocketPeerVerifier(dylib, desktop_cdhash=risky_hash, worker_cdhash=paths[1][1])

        def observe(label, binary, *, inject=False, peer_verifier=verifier):
            path, marker = root / (label + '.sock'), root / (label + '.marker')
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(str(path)); listener.listen(1); listener.settimeout(10)
                env = dict(os.environ)
                if inject:
                    env.update(DYLD_INSERT_LIBRARIES=str(injected_library), PROBE_MARKER=str(marker))
                process = subprocess.Popen([str(binary), str(path)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    connection, _ = listener.accept()
                    with connection:
                        result = (peer_verifier(connection, 'desktop'), peer_verifier(connection, 'worker'))
                        connection.sendall(b'x')
                    process.communicate(timeout=10)
                    if process.returncode != 0:
                        raise RuntimeError('fixture_client_failed')
                    return result, marker.exists()
                finally:
                    if process.poll() is None:
                        process.kill(); process.communicate(timeout=5)

        assert observe('desktop', paths[0][0]) == ((True, False), False)
        # A copied executable is the same identity; the hash does not pin path.
        assert observe('identical-copy', clone) == ((True, False), False)
        assert observe('worker', paths[1][0]) == ((False, True), False)
        assert observe('hardened-injection', paths[0][0], inject=True) == ((True, False), False)
        # Control confirms the synthetic dylib would be loaded without runtime.
        # Even a correctly pinned unhardened binary is rejected.
        assert observe('unhardened-control', plain, inject=True, peer_verifier=plain_verifier) == ((False, False), True)
        # The exact pinned hash also cannot override unsafe runtime entitlements.
        assert observe('risky-entitlement', risky, peer_verifier=risky_verifier) == ((False, False), False)
        print('pinned roles, copy, mismatched hash, hardened and entitlement controls passed')
    print('temporary fixture removed')


if __name__ == '__main__':
    main()
