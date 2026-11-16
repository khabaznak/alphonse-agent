"""Unregistered storage primitive for owner knowledge provider authorization.

This module does not authenticate management callers or identify a provider.
Only a trusted owner-authenticated management boundary may issue/revoke grants,
and provider identity must be derived from trusted runtime configuration, never
from a model or request. The primitive is not wired into the running loop.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
import sqlite3
from uuid import uuid4

from alphonse.agent_v2.database import connect_database


@dataclass(frozen=True)
class ProviderIdentity:
    """Non-secret identity of the exact configured disclosure destination.

    The host is responsible for defining stable, unambiguous account and
    endpoint identifiers and for changing these values when routing changes.
    Credentials must never be supplied as identifiers.
    """
    provider: str
    endpoint: str = field(repr=False)
    account: str = field(repr=False)
    model: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value or value != value.strip() or len(value) > 512
               for value in (self.provider, self.endpoint, self.account, self.model)):
            raise ValueError('knowledge_disclosure_provider_invalid')


@dataclass(frozen=True, repr=False)
class DisclosureBinding:
    grant_id: str
    owner: str
    provider: ProviderIdentity


def _owner_locked(conn: sqlite3.Connection, owner: str) -> bool:
    if not isinstance(owner, str) or not owner or owner != owner.strip():
        return False
    row = conn.execute('''SELECT 1 FROM v2_user_settings s JOIN v2_users u ON u.user_id=s.value
        WHERE s.key='admin_user_id' AND s.value=? AND u.role IN ('admin','owner') AND u.is_active=1''',
                       (owner,)).fetchone()
    return row is not None


def disclosure_binding_locked(conn: sqlite3.Connection, owner: str,
                              provider: ProviderIdentity) -> DisclosureBinding | None:
    """Read authorization in the caller's claim-fenced write transaction.

    A live owner claim, cancellation and trusted provider selection remain the
    caller's responsibility. The caller must begin a write transaction on the
    authoritative DB before this check; merely opening a deferred transaction
    does not fence a concurrent revocation. A read is a point-in-time decision
    and cannot retract information already sent to a provider.
    """
    if not isinstance(conn, sqlite3.Connection) or not conn.in_transaction or type(provider) is not ProviderIdentity:
        return None
    try:
        if not _owner_locked(conn, owner):
            return None
        row = conn.execute('''SELECT grant_id FROM owner_knowledge_disclosure_grants
            WHERE owner=? AND provider=? AND endpoint=? AND account=? AND model=?''',
                           (owner, provider.provider, provider.endpoint, provider.account, provider.model)).fetchone()
        if (row is None or not isinstance(row[0], str) or len(row[0]) != 32
                or any(char not in '0123456789abcdef' for char in row[0])):
            return None
        return DisclosureBinding(row[0], owner, provider)
    except (sqlite3.Error, TypeError, ValueError):
        return None


def disclosure_matches_locked(conn: sqlite3.Connection, binding: DisclosureBinding) -> bool:
    if type(binding) is not DisclosureBinding:
        return False
    return disclosure_binding_locked(conn, binding.owner, binding.provider) == binding


class OwnerKnowledgeDisclosureStore:
    """Storage only; methods require an already-authenticated management caller."""

    def __init__(self, db_path: str) -> None:
        if not isinstance(db_path, str) or not db_path or db_path == ':memory:':
            raise ValueError('knowledge_disclosure_database_invalid')
        self.db_path = db_path
        with closing(connect_database(db_path)) as conn, conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS owner_knowledge_disclosure_grants (
                owner TEXT PRIMARY KEY, provider TEXT NOT NULL, endpoint TEXT NOT NULL,
                account TEXT NOT NULL, model TEXT NOT NULL, grant_id TEXT NOT NULL UNIQUE
            ) STRICT''')

    def issue(self, owner: str, provider: ProviderIdentity) -> DisclosureBinding:
        """Replace the owner's grant after explicit, authenticated authorization."""
        if type(provider) is not ProviderIdentity:
            raise ValueError('knowledge_disclosure_provider_invalid')
        with closing(connect_database(self.db_path)) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            if not _owner_locked(conn, owner):
                raise PermissionError('knowledge_disclosure_owner_unavailable')
            grant_id = uuid4().hex
            conn.execute('''INSERT INTO owner_knowledge_disclosure_grants VALUES (?,?,?,?,?,?)
                ON CONFLICT(owner) DO UPDATE SET provider=excluded.provider, endpoint=excluded.endpoint,
                account=excluded.account, model=excluded.model, grant_id=excluded.grant_id''',
                         (owner, provider.provider, provider.endpoint, provider.account, provider.model, grant_id))
            return DisclosureBinding(grant_id, owner, provider)

    def revoke(self, owner: str) -> None:
        """Revoke all owner disclosure grants without authorizing a replacement."""
        with closing(connect_database(self.db_path)) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            if not _owner_locked(conn, owner):
                raise PermissionError('knowledge_disclosure_owner_unavailable')
            conn.execute('DELETE FROM owner_knowledge_disclosure_grants WHERE owner=?', (owner,))
