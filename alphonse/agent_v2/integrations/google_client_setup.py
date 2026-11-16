"""Import a Google Desktop client secret into the configured credential vault.

Run locally, never through IPC or model tools. The downloaded file is left in
place; no secret value is printed or accepted as a command-line argument.
"""
from __future__ import annotations

import json
import argparse
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
    parser = argparse.ArgumentParser(description='Import a downloaded Google Desktop client to the configured vault.')
    parser.add_argument('path', nargs='?', help='Downloaded Google Desktop client JSON (not deleted)')
    parser.add_argument('--persist', action='store_true', help='Save the non-secret client config in the owner database')
    parser.add_argument('--persist-existing', action='store_true', help='Save complete environment config using its existing vault reference')
    parser.add_argument('--replace', action='store_true', help='Explicitly replace a saved Google client config')
    args = parser.parse_args()
    if ((args.replace and not (args.persist or args.persist_existing))
            or (args.persist_existing and (args.persist or args.path))
            or (not args.persist_existing and not args.path)):
        print('Choose a downloaded client file or --persist-existing; --replace requires a persist mode.', file=sys.stderr)
        return 2
    vault = CredentialVault()
    reference = None
    try:
        client_id = os.environ.get('ALPHONSE_GOOGLE_OAUTH_CLIENT_ID', '')
        if args.persist_existing:
            reference = os.environ.get('ALPHONSE_GOOGLE_OAUTH_CLIENT_SECRET_REF', '')
            credential = vault.get(reference, 'oauth-client:google:' + client_id, 1)
            if set(credential) != {'client_secret'}:
                raise ValueError()
        else:
            reference = import_desktop_client(args.path, client_id, vault)
        if args.persist or args.persist_existing:
            from alphonse.agent_v2.database import default_database_path
            from alphonse.agent_v2.integrations.google_client_config import save_google_config
            from alphonse.agent_v2.integrations.oauth import ClientConfig
            config = ClientConfig('google', client_id, os.environ.get('ALPHONSE_GOOGLE_OAUTH_REDIRECT_URI', ''), reference)
            save_google_config(default_database_path(), config, replace=args.replace)
    except Exception:
        # A database commit can succeed even if the caller sees an error. Never
        # delete a fresh credential here: it may already be the saved reference.
        print('Google Desktop client import failed. Check the configured client ID, downloaded file and credential storage.', file=sys.stderr)
        return 1
    if args.persist or args.persist_existing:
        print('Google client configuration saved. Restart Alphonse. Google OAuth environment overrides take precedence over this saved configuration.')
    else:
        print('Google Desktop client stored in the configured credential vault.')
        print('Set ALPHONSE_GOOGLE_OAUTH_CLIENT_SECRET_REF to this reference and restart Alphonse:')
        print(reference)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
