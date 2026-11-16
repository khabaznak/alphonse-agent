"""Synthetic claim fencing; no production queue migration is performed."""
from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.messages.claim_fence import (
    ClaimFenceUnavailable, ClaimReceipt, ClaimSuperseded, fenced_claim_transaction, new_claim_generation,
)
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def q(tmp_path):
    path = tmp_path / 'queue.sqlite'
    queue = SQLiteMessageQueue(path, lease_owner='worker')
    queue.enqueue(CoreMessage(prompt='synthetic', user='alice', timestamp=NOW), message_id='msg')
    # Future queue migration simulated ONLY in this disposable database.
    with queue._connect() as conn:
        conn.execute("ALTER TABLE v2_inbound_messages ADD COLUMN claim_generation TEXT NOT NULL DEFAULT ''")
        conn.execute('CREATE TABLE effects (value TEXT NOT NULL)')
        conn.execute("UPDATE v2_inbound_messages SET status='processing', lease_owner='worker', "
                     "lease_expires_at=?, claim_generation=? WHERE message_id='msg'",
                     ((NOW + timedelta(minutes=1)).isoformat(), 'a' * 32))
    return path, ClaimReceipt('msg', 'alice', 'worker', 'a' * 32)


def connect(path):
    return sqlite3.connect(path, timeout=2)


def test_live_receipt_commits_only_same_database_work(q):
    path, claim = q
    with connect(path) as conn:
        with fenced_claim_transaction(conn, claim, clock=lambda: NOW) as transaction:
            transaction.execute("INSERT INTO effects VALUES ('one')")
        assert conn.execute('SELECT COUNT(*) FROM effects').fetchone()[0] == 1


def test_same_owner_reclaim_or_reinsert_rotates_generation_and_blocks_stale_worker(q):
    path, claim = q
    with connect(path) as conn:
        conn.execute("UPDATE v2_inbound_messages SET claim_generation=?, attempt_count=1 WHERE message_id='msg'", ('b' * 32,))
        conn.commit()
        with pytest.raises(ClaimSuperseded):
            with fenced_claim_transaction(conn, claim, clock=lambda: NOW):
                pytest.fail('stale claim entered transaction')
        assert conn.execute('SELECT COUNT(*) FROM effects').fetchone()[0] == 0
        conn.execute("DELETE FROM v2_inbound_messages WHERE message_id='msg'")
        conn.commit()
    # A new row for the same ID and owner uses a new generation: no ABA.
    queue = SQLiteMessageQueue(path, lease_owner='worker')
    queue.enqueue(CoreMessage(prompt='synthetic-new', user='alice', timestamp=NOW), message_id='msg')
    with connect(path) as conn:
        conn.execute("UPDATE v2_inbound_messages SET status='processing', lease_owner='worker', "
                     "lease_expires_at=?, claim_generation=? WHERE message_id='msg'",
                     ((NOW + timedelta(minutes=1)).isoformat(), 'c' * 32))
        conn.commit()
        with pytest.raises(ClaimSuperseded):
            with fenced_claim_transaction(conn, claim, clock=lambda: NOW):
                pass


@pytest.mark.parametrize('field,value', [('user_id', 'bob'), ('lease_owner', 'other'), ('status', 'pending')])
def test_changed_actor_owner_or_status_rejects_claim(q, field, value):
    path, claim = q
    with connect(path) as conn:
        conn.execute(f"UPDATE v2_inbound_messages SET {field}=? WHERE message_id='msg'", (value,))
        conn.commit()
        with pytest.raises(ClaimSuperseded):
            with fenced_claim_transaction(conn, claim, clock=lambda: NOW):
                pytest.fail('superseded claim entered')


def test_expiry_or_clock_rollback_during_transaction_rolls_back_writes(q):
    path, claim = q
    with connect(path) as conn:
        clock = iter((NOW, NOW + timedelta(minutes=1)))
        with pytest.raises(ClaimSuperseded, match='expired'):
            with fenced_claim_transaction(conn, claim, clock=lambda: next(clock)) as transaction:
                transaction.execute("INSERT INTO effects VALUES ('must roll back')")
        assert conn.execute('SELECT COUNT(*) FROM effects').fetchone()[0] == 0
        clock = iter((NOW, NOW - timedelta(seconds=1)))
        with pytest.raises(ClaimFenceUnavailable, match='backwards'):
            with fenced_claim_transaction(conn, claim, clock=lambda: next(clock)) as transaction:
                transaction.execute("INSERT INTO effects VALUES ('must also roll back')")
        assert conn.execute('SELECT COUNT(*) FROM effects').fetchone()[0] == 0


def test_claim_row_mutation_rolls_back_transaction(q):
    path, claim = q
    with connect(path) as conn:
        with pytest.raises(ClaimSuperseded):
            with fenced_claim_transaction(conn, claim, clock=lambda: NOW) as transaction:
                transaction.execute("INSERT INTO effects VALUES ('must roll back')")
                transaction.execute("UPDATE v2_inbound_messages SET status='pending' WHERE message_id='msg'")
        assert conn.execute('SELECT COUNT(*) FROM effects').fetchone()[0] == 0
        assert conn.execute("SELECT status FROM v2_inbound_messages WHERE message_id='msg'").fetchone()[0] == 'processing'


@pytest.mark.parametrize('field,value', [
    ('lease_expires_at', (NOW + timedelta(hours=1)).isoformat()),
    ('last_error', 'rewritten'),
])
def test_claim_row_changes_that_keep_claim_live_are_rolled_back(q, field, value):
    path, claim = q
    with connect(path) as conn:
        before = conn.execute(f'SELECT {field} FROM v2_inbound_messages WHERE message_id=?', ('msg',)).fetchone()[0]
        with pytest.raises(ClaimFenceUnavailable, match='claim_row_modified_by_caller'):
            with fenced_claim_transaction(conn, claim, clock=lambda: NOW) as transaction:
                transaction.execute("INSERT INTO effects VALUES ('must roll back')")
                transaction.execute(f'UPDATE v2_inbound_messages SET {field}=? WHERE message_id=?', (value, 'msg'))
        assert conn.execute('SELECT COUNT(*) FROM effects').fetchone()[0] == 0
        assert conn.execute(f'SELECT {field} FROM v2_inbound_messages WHERE message_id=?', ('msg',)).fetchone()[0] == before


def test_competing_reclaim_cannot_change_claim_inside_fenced_transaction(q):
    path, claim = q
    with connect(path) as first, sqlite3.connect(path, timeout=0.01) as second:
        with fenced_claim_transaction(first, claim, clock=lambda: NOW) as transaction:
            transaction.execute("INSERT INTO effects VALUES ('checkpoint')")
            with pytest.raises(sqlite3.OperationalError, match='locked'):
                second.execute("UPDATE v2_inbound_messages SET claim_generation=? WHERE message_id='msg'", ('b' * 32,))
            second.rollback()
        second.execute("UPDATE v2_inbound_messages SET claim_generation=? WHERE message_id='msg'", ('b' * 32,))
        second.commit()
        with pytest.raises(ClaimSuperseded):
            with fenced_claim_transaction(first, claim, clock=lambda: NOW):
                pass
        assert first.execute('SELECT COUNT(*) FROM effects').fetchone()[0] == 1


def test_legacy_schema_missing_generation_fails_closed(tmp_path):
    path = tmp_path / 'legacy.sqlite'
    queue = SQLiteMessageQueue(path, lease_owner='worker')
    queue.enqueue(CoreMessage(prompt='synthetic', user='alice', timestamp=NOW), message_id='msg')
    with connect(path) as conn:
        with pytest.raises(ClaimFenceUnavailable, match='storage_unavailable'):
            with fenced_claim_transaction(conn, ClaimReceipt('msg', 'alice', 'worker', 'a' * 32), clock=lambda: NOW):
                pytest.fail('legacy schema has no generation')


def test_temp_table_cannot_shadow_main_claim_table(q):
    path, claim = q
    with connect(path) as conn:
        conn.execute("UPDATE main.v2_inbound_messages SET claim_generation=? WHERE message_id='msg'", ('b' * 32,))
        conn.execute('CREATE TEMP TABLE v2_inbound_messages(message_id TEXT,status TEXT,user_id TEXT,lease_owner TEXT,claim_generation TEXT,lease_expires_at TEXT)')
        conn.execute('INSERT INTO temp.v2_inbound_messages VALUES (?,?,?,?,?,?)',
                     ('msg', 'processing', 'alice', 'worker', 'a' * 32, (NOW + timedelta(minutes=1)).isoformat()))
        conn.commit()
        with pytest.raises(ClaimSuperseded):
            with fenced_claim_transaction(conn, claim, clock=lambda: NOW):
                pytest.fail('temp shadow must not establish a live main-DB claim')


def test_missing_or_naive_expiry_fails_closed(q):
    path, claim = q
    with connect(path) as conn:
        for expiry in ('', '2026-01-01T00:01:00'):
            conn.execute("UPDATE v2_inbound_messages SET lease_expires_at=? WHERE message_id='msg'", (expiry,))
            conn.commit()
            with pytest.raises(ClaimFenceUnavailable, match='expiry_invalid'):
                with fenced_claim_transaction(conn, claim, clock=lambda: NOW):
                    pass


def test_token_is_opaque_fresh_and_hidden_from_repr():
    first, second = new_claim_generation(), new_claim_generation()
    assert len(first) == 32 and first != second
    assert first not in repr(ClaimReceipt('msg', 'alice', 'worker', first))
    with pytest.raises(ValueError):
        ClaimReceipt('msg', 'alice', 'worker', 'bad')
