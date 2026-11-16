"""Validated owner-local Google Desktop configuration; never stores secrets."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import stat
from urllib.parse import urlsplit

from .oauth import ClientConfig

_KEY = 'google_desktop_oauth_v1'
_ID = 'ALPHONSE_GOOGLE_OAUTH_CLIENT_ID'
_REDIRECT = 'ALPHONSE_GOOGLE_OAUTH_REDIRECT_URI'
_REF = 'ALPHONSE_GOOGLE_OAUTH_CLIENT_SECRET_REF'


def _validate(client_id: str, redirect: str, reference: str) -> ClientConfig:
    try:
        config = ClientConfig('google', client_id, redirect, reference)
        uri = urlsplit(redirect)
        if uri.scheme != 'http' or uri.hostname != '127.0.0.1' or not uri.port or not reference:
            raise ValueError()
        return config
    except Exception:
        raise ValueError('google_client_configuration_invalid') from None


def resolve_google_config(user_store, environ) -> ClientConfig | None:
    """Environment, when present, overrides the entire stored configuration.

    A partial/invalid environment never borrows a vault reference from storage.
    Legacy environment-only configuration remains supported, including without
    an optional client credential. Stored config is strictly validated.
    """
    if any(key in environ for key in (_ID, _REDIRECT, _REF)):
        client_id, redirect, reference = (environ.get(key, '') for key in (_ID, _REDIRECT, _REF))
        if not client_id and not redirect and not reference:
            return None
        return ClientConfig('google', client_id, redirect, reference)
    with user_store._connect() as conn:
        row = conn.execute('''SELECT c.value, (SELECT u.user_id FROM v2_users u JOIN v2_user_settings s
            ON s.key='admin_user_id' AND s.value=u.user_id WHERE u.role IN ('admin','owner') AND u.is_active=1)
            FROM v2_user_settings c WHERE c.key=?''', (_KEY,)).fetchone()
    if row is None:
        return None
    try:
        value = json.loads(row[0])
        if (not isinstance(value, dict) or set(value) != {'schema_version', 'owner', 'client_id', 'redirect_uri', 'credential_ref'}
                or type(value['schema_version']) is not int or value['schema_version'] != 1
                or not isinstance(row[1], str) or value['owner'] != row[1]):
            raise ValueError()
        return _validate(value['client_id'], value['redirect_uri'], value['credential_ref'])
    except Exception:
        raise ValueError('google_client_configuration_invalid') from None


def save_google_config(db_path: str | Path, config: ClientConfig, *, replace: bool = False) -> None:
    """Persist for the active owner via a trusted same-UID local setup process.

    This is not a remote authentication surface. Refuse replacement by default;
    do not create a database or infer an owner when no unique active owner exists.
    """
    if type(replace) is not bool or type(config) is not ClientConfig or config.provider != 'google':
        raise ValueError('google_client_configuration_invalid')
    config = _validate(config.client_id, config.redirect_uri, config.client_secret_ref)
    try:
        path = Path(db_path).expanduser().resolve(strict=True)
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError()
        with closing(sqlite3.connect(path.as_uri() + '?mode=rw', uri=True, timeout=5)) as conn:
            conn.execute('BEGIN IMMEDIATE')
            owners = conn.execute('''SELECT u.user_id FROM v2_users u JOIN v2_user_settings s
                ON s.key='admin_user_id' AND s.value=u.user_id
                WHERE u.role IN ('admin','owner') AND u.is_active=1''').fetchall()
            if len(owners) != 1:
                raise ValueError()
            if conn.execute('SELECT 1 FROM v2_user_settings WHERE key=?', (_KEY,)).fetchone() and not replace:
                raise FileExistsError('google_client_configuration_exists')
            value = json.dumps({'schema_version': 1, 'owner': owners[0][0], 'client_id': config.client_id,
                'redirect_uri': config.redirect_uri, 'credential_ref': config.client_secret_ref}, sort_keys=True)
            conn.execute('''INSERT INTO v2_user_settings(key,value,updated_at) VALUES(?,?,?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at''',
                         (_KEY, value, datetime.now(timezone.utc).isoformat()))
            conn.commit()
    except FileExistsError:
        raise
    except Exception:
        raise ValueError('google_client_configuration_unavailable') from None
