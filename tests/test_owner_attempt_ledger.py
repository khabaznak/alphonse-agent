import sqlite3

import pytest

from alphonse.agent_v2.core.intelligence.simple_loop.owner_attempt_ledger import (
    AttemptLedgerError, begin, finish, install, read_verified, unresolved_attempts,
)


@pytest.fixture
def conn():
    db = sqlite3.connect(':memory:')
    install(db)
    db.commit()
    yield db
    db.close()


def start(conn, attempt='a', owner='owner', task='task'):
    begin(conn, owner_id=owner, task_id=task, attempt_id=attempt,
          tool_id='native.web_search', arguments={'query': 'synthetic query'})


def read(conn, owner='owner', task='task'):
    return read_verified(conn, owner_id=owner, task_id=task)


def test_attempt_without_outcome_stays_unknown_across_reopen(tmp_path):
    path = tmp_path / 'ledger.db'
    first = sqlite3.connect(path)
    install(first)
    first.execute('BEGIN')
    start(first)
    first.commit()
    first.close()
    second = sqlite3.connect(path)
    events = read(second)
    assert unresolved_attempts(events) == ['a']
    assert 'synthetic query' not in repr(events)
    second.close()


def test_outcome_is_append_only_and_can_be_recorded_once(conn):
    conn.execute('BEGIN')
    start(conn)
    finish(conn, owner_id='owner', task_id='task', attempt_id='a', outcome='failed')
    assert unresolved_attempts(read(conn)) == []
    with pytest.raises(AttemptLedgerError):
        finish(conn, owner_id='owner', task_id='task', attempt_id='a', outcome='success')
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE v2_owner_attempt_ledger SET event_json='{}'")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute('DELETE FROM v2_owner_attempt_ledger')


def test_transaction_rollback_does_not_record_attempt(conn):
    conn.execute('BEGIN')
    start(conn)
    conn.rollback()
    assert read(conn) == []
    with pytest.raises(AttemptLedgerError, match='transaction_required'):
        start(conn)


def test_isolation_and_duplicate_attempt_ids(conn):
    conn.execute('BEGIN')
    start(conn)
    start(conn, owner='other')
    start(conn, task='another')
    with pytest.raises(AttemptLedgerError, match='duplicate'):
        start(conn)
    assert len(read(conn)) == len(read(conn, owner='other')) == len(read(conn, task='another')) == 1


def test_tampered_event_fails_integrity_check(conn):
    conn.execute('BEGIN')
    start(conn)
    conn.execute('DROP TRIGGER v2_owner_attempt_ledger_no_update')
    conn.execute("UPDATE v2_owner_attempt_ledger SET event_json='{}'")
    with pytest.raises(AttemptLedgerError):
        read(conn)


def test_broken_sequence_and_unknown_outcome_rejected(conn):
    conn.execute('BEGIN')
    start(conn)
    with pytest.raises(AttemptLedgerError):
        finish(conn, owner_id='owner', task_id='task', attempt_id='a', outcome='unknown')
    assert unresolved_attempts(read(conn)) == ['a']
    conn.execute('DROP TRIGGER v2_owner_attempt_ledger_no_update')
    conn.execute('UPDATE v2_owner_attempt_ledger SET sequence=3')
    with pytest.raises(AttemptLedgerError, match='corrupt'):
        read(conn)
