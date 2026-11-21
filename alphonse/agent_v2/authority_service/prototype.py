"""Prototype service-owned request, claim, and action-intent ledger.

This module is not an authorization boundary by itself. A real deployment must
run it under a separately protected OS identity and provide a transport that
authenticates kernel-provided peer identity. No production adapter is included.
It does not execute Bash, authorize household resources, or verify effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import secrets
import sqlite3
import time
from typing import Callable, Literal


@dataclass(frozen=True)
class AuthenticatedPeer:
    """Identity returned by a trusted transport, never parsed from a request."""

    role: Literal['desktop', 'worker']
    actor: str
    client_id: str


class AuthorityDenied(ValueError):
    """The requested authority transition could not be established."""


def _text(value, *, maximum):
    if not isinstance(value, str) or not value.strip() or len(value.encode('utf-8')) > maximum:
        raise AuthorityDenied('authority_invalid_input')
    return value


def _hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _action_digest(action):
    if not isinstance(action, dict):
        raise AuthorityDenied('authority_invalid_action')
    try:
        text = json.dumps(action, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        raise AuthorityDenied('authority_invalid_action') from None
    if len(text.encode('utf-8')) > 65536:
        raise AuthorityDenied('authority_invalid_action')
    return _hash(text)


class AuthorityService:
    """Durable prototype with a mandatory server-side peer authenticator.

    ``authenticate`` is called with an opaque transport connection and must
    derive identity from the OS, not from client-provided JSON. The injected
    clock and authenticator are service dependencies, not request parameters.
    The service process alone must own its SQLite database in a real system.
    """

    def __init__(self, path, *, authenticate: Callable[[object], AuthenticatedPeer], clock=time.time):
        if not callable(authenticate) or not callable(clock):
            raise ValueError('authority_transport_required')
        path = Path(path)
        if not path.is_absolute() or str(path) == ':memory:':
            raise ValueError('authority_persistent_path_required')
        self.path = str(path)
        self.authenticate = authenticate
        self.clock = clock
        with self._connect() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, idempotency_key TEXT NOT NULL,
                    goal_sha256 TEXT NOT NULL, status TEXT NOT NULL
                        CHECK(status IN ('pending','claimed','cancelled','finished')),
                    generation INTEGER NOT NULL DEFAULT 0, worker TEXT NOT NULL DEFAULT '',
                    token_sha256 TEXT NOT NULL DEFAULT '', lease_until REAL NOT NULL DEFAULT 0,
                    UNIQUE(owner,idempotency_key)
                ) STRICT;
                CREATE TABLE IF NOT EXISTS actions (
                    id TEXT PRIMARY KEY, request_id TEXT NOT NULL REFERENCES requests(id),
                    idempotency_key TEXT NOT NULL, action_sha256 TEXT NOT NULL, state TEXT NOT NULL
                        CHECK(state IN ('reserved','started','return_reported')),
                    return_sha256 TEXT NOT NULL DEFAULT '', UNIQUE(request_id,idempotency_key)
                ) STRICT;
                CREATE TABLE IF NOT EXISTS bash_capabilities (
                    owner TEXT PRIMARY KEY, enabled INTEGER NOT NULL CHECK(enabled IN (0,1)),
                    generation INTEGER NOT NULL CHECK(generation >= 0)
                ) STRICT;
                CREATE TABLE IF NOT EXISTS bash_claims (
                    request_id TEXT PRIMARY KEY REFERENCES requests(id),
                    generation INTEGER NOT NULL CHECK(generation > 0)
                ) STRICT;
            ''')

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        try:
            yield conn
        finally:
            conn.close()

    def _now(self):
        now = self.clock()
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            raise AuthorityDenied('authority_clock_unavailable')
        return float(now)

    def _peer(self, connection, role):
        try:
            peer = self.authenticate(connection)
        except Exception:
            raise AuthorityDenied('authority_peer_denied') from None
        if (type(peer) is not AuthenticatedPeer or peer.role != role
                or not isinstance(peer.actor, str) or not peer.actor.strip() or len(peer.actor) > 256
                or not isinstance(peer.client_id, str) or not peer.client_id.strip() or len(peer.client_id) > 256):
            raise AuthorityDenied('authority_peer_denied')
        return peer

    def submit(self, connection, *, goal, idempotency_key):
        """Bind exact owner input; actor identity comes only from the transport."""
        peer = self._peer(connection, 'desktop')
        goal = _text(goal, maximum=65536)
        if not isinstance(idempotency_key, str) or re.fullmatch(r'[0-9a-f]{32}', idempotency_key) is None:
            raise AuthorityDenied('authority_invalid_input')
        digest = _hash(goal)
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            existing = conn.execute('SELECT id,goal_sha256 FROM requests WHERE owner=? AND idempotency_key=?',
                                    (peer.actor, idempotency_key)).fetchone()
            if existing:
                if existing['goal_sha256'] != digest:
                    raise AuthorityDenied('authority_idempotency_conflict')
                conn.commit()
                return {'request_id': existing['id']}
            request_id = secrets.token_hex(16)
            conn.execute('INSERT INTO requests(id,owner,idempotency_key,goal_sha256,status) VALUES (?,?,?,?,?)',
                         (request_id, peer.actor, idempotency_key, digest, 'pending'))
            conn.commit()
            return {'request_id': request_id}

    def bash_capability(self, connection):
        """Return the authenticated owner's persistent capability, defaulting off."""
        peer = self._peer(connection, 'desktop')
        with self._connect() as conn:
            row = conn.execute('SELECT enabled,generation FROM bash_capabilities WHERE owner=?',
                               (peer.actor,)).fetchone()
            return {'enabled': bool(row['enabled']) if row else False,
                    'generation': row['generation'] if row else 0}

    def _set_bash_capability(self, connection, enabled, expected_generation):
        peer = self._peer(connection, 'desktop')
        if type(expected_generation) is not int or not 0 <= expected_generation < 2**63 - 1:
            raise AuthorityDenied('authority_invalid_input')
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT enabled,generation FROM bash_capabilities WHERE owner=?',
                               (peer.actor,)).fetchone()
            old_enabled = bool(row['enabled']) if row else False
            old_generation = row['generation'] if row else 0
            if old_generation != expected_generation:
                raise AuthorityDenied('authority_capability_changed')
            # An explicit disable invalidates pending grants even if already
            # disabled, so an in-flight setup prompt cannot undo revocation.
            generation = old_generation + (old_enabled != enabled or not enabled)
            conn.execute('''INSERT INTO bash_capabilities(owner,enabled,generation) VALUES (?,?,?)
                ON CONFLICT(owner) DO UPDATE SET enabled=excluded.enabled,generation=excluded.generation''',
                (peer.actor, int(enabled), generation))
            conn.commit()
            return {'enabled': enabled, 'generation': generation}

    def grant_bash(self, connection, *, expected_generation):
        return self._set_bash_capability(connection, True, expected_generation)

    def revoke_bash(self, connection, *, expected_generation):
        return self._set_bash_capability(connection, False, expected_generation)

    def lookup_request(self, connection, *, message_id, goal):
        """Find an owner request after a lost submit reply without creating it."""
        peer = self._peer(connection, 'desktop')
        _text(goal, maximum=65536)
        if not isinstance(message_id, str) or re.fullmatch(r'[0-9a-f]{32}', message_id) is None:
            raise AuthorityDenied('authority_request_denied')
        with self._connect() as conn:
            conn.execute('BEGIN')
            row = conn.execute('SELECT * FROM requests WHERE owner=? AND idempotency_key=?',
                               (peer.actor, message_id)).fetchone()
            if row is None:
                conn.commit()
                return {'actor': peer.actor, 'found': False}
            if not hmac.compare_digest(row['goal_sha256'], _hash(goal)):
                raise AuthorityDenied('authority_request_denied')
            started = conn.execute("SELECT COUNT(*) FROM actions WHERE request_id=? AND state IN ('started','return_reported')",
                                   (row['id'],)).fetchone()[0]
            conn.commit()
            return {'actor': peer.actor, 'found': True, 'request_id': row['id'],
                    'status': row['status'], 'started_actions': started}

    def inspect_request(self, connection, *, request_id, message_id, goal):
        """Inspect a Desktop request only with the exact owner, origin ID and goal."""
        peer = self._peer(connection, 'desktop')
        _text(request_id, maximum=64)
        _text(goal, maximum=65536)
        if not isinstance(message_id, str) or re.fullmatch(r'[0-9a-f]{32}', message_id) is None:
            raise AuthorityDenied('authority_request_denied')
        with self._connect() as conn:
            conn.execute('BEGIN')
            row = conn.execute('SELECT * FROM requests WHERE id=? AND owner=?', (request_id, peer.actor)).fetchone()
            if (row is None or row['idempotency_key'] != message_id
                    or not hmac.compare_digest(row['goal_sha256'], _hash(goal))):
                raise AuthorityDenied('authority_request_denied')
            started = conn.execute("SELECT COUNT(*) FROM actions WHERE request_id=? AND state IN ('started','return_reported')",
                                   (request_id,)).fetchone()[0]
            conn.commit()
            return {'actor': peer.actor, 'status': row['status'], 'started_actions': started}

    def claim(self, connection, *, request_id, message_id, goal, expected_actor, lease_seconds=60):
        """Issue one short-lived worker claim; never silently reclaim one."""
        peer = self._peer(connection, 'worker')
        # The service-derived peer remains authoritative. This caller-supplied
        # value can only narrow access, preventing a mismatched task from
        # consuming another owner's pending claim; it cannot grant identity.
        if not isinstance(expected_actor, str) or peer.actor != expected_actor:
            raise AuthorityDenied('authority_claim_denied')
        _text(request_id, maximum=64)
        _text(goal, maximum=65536)
        if not isinstance(message_id, str) or re.fullmatch(r'[0-9a-f]{32}', message_id) is None:
            raise AuthorityDenied('authority_claim_denied')
        if (isinstance(lease_seconds, bool) or not isinstance(lease_seconds, (int, float))
                or not math.isfinite(lease_seconds) or not 0 < lease_seconds <= 300):
            raise AuthorityDenied('authority_invalid_lease')
        now = self._now()
        token = secrets.token_hex(32)
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT * FROM requests WHERE id=? AND owner=?', (request_id, peer.actor)).fetchone()
            if (row is None or row['status'] != 'pending' or row['idempotency_key'] != message_id
                    or not hmac.compare_digest(row['goal_sha256'], _hash(goal))):
                raise AuthorityDenied('authority_claim_denied')
            generation = row['generation'] + 1
            conn.execute('''UPDATE requests SET status='claimed',generation=?,worker=?,token_sha256=?,lease_until=?
                WHERE id=?''', (generation, peer.client_id, _hash(token), now + lease_seconds, request_id))
            conn.commit()
            return {'request_id': request_id, 'generation': generation, 'token': token}

    def _claim_locked(self, conn, peer, claim):
        if (not isinstance(claim, dict) or set(claim) != {'request_id', 'generation', 'token'}
                or not isinstance(claim['request_id'], str) or re.fullmatch(r'[0-9a-f]{32}', claim['request_id']) is None
                or type(claim['generation']) is not int or claim['generation'] < 1
                or not isinstance(claim['token'], str) or re.fullmatch(r'[0-9a-f]{64}', claim['token']) is None):
            raise AuthorityDenied('authority_claim_denied')
        row = conn.execute('SELECT * FROM requests WHERE id=? AND owner=?',
                           (claim['request_id'], peer.actor)).fetchone()
        if (row is None or row['status'] != 'claimed' or row['worker'] != peer.client_id
                or row['generation'] != claim['generation'] or row['lease_until'] <= self._now()
                or not hmac.compare_digest(row['token_sha256'], _hash(claim['token']))):
            raise AuthorityDenied('authority_claim_denied')
        bound = conn.execute('SELECT generation FROM bash_claims WHERE request_id=?',
                             (claim['request_id'],)).fetchone()
        if bound:
            capability = conn.execute('SELECT enabled,generation FROM bash_capabilities WHERE owner=?',
                                      (peer.actor,)).fetchone()
            if not capability or not capability['enabled'] or capability['generation'] != bound['generation']:
                raise AuthorityDenied('authority_bash_revoked')
        return row

    def reserve_action(self, connection, *, claim, action, idempotency_key):
        """Bind an exact action to a live claim; this does not execute it.

        A stable action key makes retries observable without preventing later,
        different actions in the same owner request.
        """
        peer = self._peer(connection, 'worker')
        digest = _action_digest(action)
        if not isinstance(idempotency_key, str) or re.fullmatch(r'[0-9a-f]{32}', idempotency_key) is None:
            raise AuthorityDenied('authority_invalid_input')
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._claim_locked(conn, peer, claim)
            if action.get('tool_id') == 'native.bash':
                capability = conn.execute('SELECT enabled,generation FROM bash_capabilities WHERE owner=?',
                                          (peer.actor,)).fetchone()
                if not capability or not capability['enabled']:
                    raise AuthorityDenied('authority_bash_disabled')
                conn.execute('INSERT OR IGNORE INTO bash_claims(request_id,generation) VALUES (?,?)',
                             (claim['request_id'], capability['generation']))
                self._claim_locked(conn, peer, claim)
            existing = conn.execute('''SELECT id,action_sha256,state FROM actions
                WHERE request_id=? AND idempotency_key=?''', (claim['request_id'], idempotency_key)).fetchone()
            if existing:
                if existing['action_sha256'] != digest:
                    raise AuthorityDenied('authority_idempotency_conflict')
                self._claim_locked(conn, peer, claim)
                conn.commit()
                return {'action_id': existing['id'], 'status': existing['state']}
            action_id = secrets.token_hex(16)
            conn.execute('''INSERT INTO actions(id,request_id,idempotency_key,action_sha256,state)
                VALUES (?,?,?,?,'reserved')''', (action_id, claim['request_id'], idempotency_key, digest))
            self._claim_locked(conn, peer, claim)
            conn.commit()
            return {'action_id': action_id, 'status': 'reserved'}

    def check_claim(self, connection, *, claim):
        """Check a worker claim without granting any new authority."""
        peer = self._peer(connection, 'worker')
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._claim_locked(conn, peer, claim)
            conn.commit()
            return True

    def check_bash_capability(self, connection, *, claim):
        """Report this live worker claim's current grant without granting an action."""
        peer = self._peer(connection, 'worker')
        with self._connect() as conn:
            conn.execute('BEGIN')
            self._claim_locked(conn, peer, claim)
            capability = conn.execute('SELECT enabled FROM bash_capabilities WHERE owner=?',
                                      (peer.actor,)).fetchone()
            conn.commit()
            return bool(capability and capability['enabled'])

    def start_action(self, connection, *, claim, action_id, action):
        """Atomically record dispatch before the worker attempts the effect."""
        peer = self._peer(connection, 'worker')
        digest = _action_digest(action)
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._claim_locked(conn, peer, claim)
            if (action.get('tool_id') == 'native.bash'
                    and not conn.execute('SELECT 1 FROM bash_claims WHERE request_id=?',
                                         (claim['request_id'],)).fetchone()):
                raise AuthorityDenied('authority_bash_disabled')
            updated = conn.execute('''UPDATE actions SET state='started'
                WHERE id=? AND request_id=? AND action_sha256=? AND state='reserved' ''',
                (action_id, claim['request_id'], digest))
            if updated.rowcount != 1:
                raise AuthorityDenied('authority_action_unavailable')
            self._claim_locked(conn, peer, claim)
            conn.commit()
            return {'status': 'started', 'effect': 'unverified'}

    def record_process_return(self, connection, *, claim, action_id, process_result):
        """Record a worker's report; verify neither the process nor its effects."""
        peer = self._peer(connection, 'worker')
        digest = _action_digest(process_result)
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._claim_locked(conn, peer, claim)
            updated = conn.execute('''UPDATE actions SET state='return_reported',return_sha256=?
                WHERE id=? AND request_id=? AND state='started' ''', (digest, action_id, claim['request_id']))
            if updated.rowcount != 1:
                raise AuthorityDenied('authority_action_unavailable')
            self._claim_locked(conn, peer, claim)
            conn.commit()
            return {'status': 'return_reported', 'effect': 'unverified'}

    def cancel(self, connection, *, request_id):
        """Revoke an owner request and report whether a dispatch started."""
        peer = self._peer(connection, 'desktop')
        _text(request_id, maximum=64)
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT status FROM requests WHERE id=? AND owner=?',
                               (request_id, peer.actor)).fetchone()
            if row is None or row['status'] not in {'pending', 'claimed'}:
                raise AuthorityDenied('authority_cancel_denied')
            started = conn.execute('''SELECT 1 FROM actions
                WHERE request_id=? AND state IN ('started','return_reported') LIMIT 1''', (request_id,)).fetchone()
            conn.execute("UPDATE requests SET status='cancelled' WHERE id=?", (request_id,))
            conn.commit()
            return {'status': 'cancelled', 'execution_may_have_started': bool(started)}
