"""Import a Google Desktop client secret into the configured credential vault.

Run locally, never through IPC or model tools. The downloaded file is left in
place; no secret value is printed or accepted as a command-line argument.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys

from alphonse.agent_v2.integrations.credentials import CredentialVault


def import_desktop_client(path: str, client_id: str, vault: CredentialVault) -> str:
    if not isinstance(client_id, str) or not client_id.strip():
        raise ValueError('google_client_id_unavailable')
    try:
        # Do not block on a FIFO before fstat rejects non-regular files.
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
        fd = os.open(Path(path).expanduser(), flags)
        with os.fdopen(fd, 'rb') as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size > 65536:
                raise ValueError()
            data = source.read(65537)
        if len(data) > 65536:
            raise ValueError()
        installed = json.loads(data)['installed']
        if (not isinstance(installed, dict) or installed.get('client_id') != client_id
                or not isinstance(installed.get('client_secret'), str) or not installed['client_secret']):
            raise ValueError()
        return vault.put('oauth-client:google:' + client_id, 1, {'client_secret': installed['client_secret']})
    except Exception:
        raise ValueError('google_client_import_unavailable') from None


def main() -> int:
    if len(sys.argv) != 2:
        print('Usage: python -m alphonse.agent_v2.integrations.google_client_setup <downloaded-desktop-client.json>', file=sys.stderr)
        return 2
    try:
        reference = import_desktop_client(sys.argv[1], os.environ.get('ALPHONSE_GOOGLE_OAUTH_CLIENT_ID', ''), CredentialVault())
    except Exception:
        print('Google Desktop client import failed. Check the configured client ID, downloaded file and credential storage.', file=sys.stderr)
        return 1
    print('Google Desktop client stored in the configured credential vault.')
    print('Set ALPHONSE_GOOGLE_OAUTH_CLIENT_SECRET_REF to this reference and restart Alphonse:')
    print(reference)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
