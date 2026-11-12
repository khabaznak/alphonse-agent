"""Unwired queue-claim fencing primitive; requires a future schema migration.

A fresh opaque generation must be persisted on every successful acquisition,
including reclaims and same-owner reacquisition. Existing queue methods do not
supply it; this module is deliberately unused by the production queue. A claim
is not authentication, access authority, or proof that an external effect did
or did not happen.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import InitVar, dataclass
from datetime import datetime, timezone
import re
import secrets
import sqlite3
from typing import Callable, Iterator


class ClaimSuperseded(RuntimeError):
    """The claim no longer identifies the live processing lease."""


class ClaimFenceUnavailable(RuntimeError):
    """The claim or its storage cannot be validated safely."""


def new_claim_generation() -> str:
    """Create a fresh 128-bit token for one queue acquisition, not a retry count."""
    return secrets.token_hex(16)


@dataclass(frozen=True, repr=False, eq=False)
class ClaimReceipt:
    """Internal queue ownership receipt; never emit or log the generation."""
    message_id: str
    user_id: str
    lease_owner: str
    claim_generation: InitVar[str] = ''

    def __post_init__(self, claim_generation: str) -> None:
        object.__setattr__(self, 'claim_generation', claim_generation)
        if (any(not isinstance(value, str) or not value.strip() for value in
                (self.message_id, self.user_id, self.lease_owner))
                or not isinstance(self.claim_generation, str)
                or re.fullmatch(r'[0-9a-f]{32}', self.claim_generation) is None):
            raise ValueError('claim_receipt_invalid')

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return (self.message_id, self.user_id, self.lease_owner, self.claim_generation) == (
            other.message_id, other.user_id, other.lease_owner, other.claim_generation)

    def __hash__(self) -> int:
        return hash((self.message_id, self.user_id, self.lease_owner, self.claim_generation))


def _time(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ClaimFenceUnavailable('claim_clock_invalid')
    return value.astimezone(timezone.utc)


def _read_claim(conn: sqlite3.Connection, receipt: ClaimReceipt, now: datetime) -> datetime:
    try:
        row = conn.execute('''SELECT status,user_id,lease_owner,claim_generation,lease_expires_at
            FROM main.v2_inbound_messages WHERE message_id=?''', (receipt.message_id,)).fetchone()
    except sqlite3.Error:
        raise ClaimFenceUnavailable('claim_storage_unavailable') from None
    if (row is None or row[0] != 'processing' or row[1] != receipt.user_id
            or row[2] != receipt.lease_owner or row[3] != receipt.claim_generation):
        raise ClaimSuperseded('claim_not_current')
    try:
        expires = _time(datetime.fromisoformat(row[4]))
    except (TypeError, ValueError, ClaimFenceUnavailable):
        raise ClaimFenceUnavailable('claim_expiry_invalid') from None
    if expires <= now:
        raise ClaimSuperseded('claim_expired')
    return expires


@contextmanager
def fenced_claim_transaction(conn: sqlite3.Connection, receipt: ClaimReceipt, *,
                             clock: Callable[[], datetime]) -> Iterator[sqlite3.Connection]:
    """Fence DB writes against the current queue claim in one immediate transaction.

    Both checks read the same locked SQLite database. The callback may update
    other tables in this transaction, but must not acknowledge, delete, release,
    renew or otherwise alter the claim row, and must never commit or roll back
    the connection. The caller is trusted code; a manual commit cannot be undone
    here. The second check rolls back if the
    lease expired while working. No external I/O, user output or memory write in
    another store is made safe by this SQLite lock. Do not hold the transaction
    during provider calls. A missing generation column is an unavailable fence,
    never an implicitly valid legacy claim.
    """
    if not isinstance(receipt, ClaimReceipt) or not callable(clock):
        raise ValueError('claim_fence_arguments_invalid')
    if conn.in_transaction:
        raise ClaimFenceUnavailable('claim_transaction_already_open')
    try:
        conn.execute('BEGIN IMMEDIATE')
        start = _time(clock())
        _read_claim(conn, receipt, start)
        # This primitive permits writes to other same-database rows only. A
        # second identity check alone would allow the callback to extend the
        # lease or rewrite the queued message while keeping the token intact.
        original_row = tuple(conn.execute(
            'SELECT * FROM main.v2_inbound_messages WHERE message_id=?', (receipt.message_id,)
        ).fetchone())
        yield conn
        if not conn.in_transaction:
            raise ClaimFenceUnavailable('claim_transaction_ended_by_caller')
        end = _time(clock())
        if end < start:
            raise ClaimFenceUnavailable('claim_clock_moved_backwards')
        _read_claim(conn, receipt, end)
        current_row = conn.execute(
            'SELECT * FROM main.v2_inbound_messages WHERE message_id=?', (receipt.message_id,)
        ).fetchone()
        if current_row is None or tuple(current_row) != original_row:
            raise ClaimFenceUnavailable('claim_row_modified_by_caller')
        conn.commit()
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
