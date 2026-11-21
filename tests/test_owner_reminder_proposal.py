import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.intelligence.simple_loop import owner_reminder_proposal as proposals


@pytest.fixture
def setup(monkeypatch):
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE v2_users (user_id TEXT PRIMARY KEY, role TEXT, is_active INTEGER);
        CREATE TABLE v2_user_settings (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO v2_users VALUES ('owner', 'owner', 1);
        INSERT INTO v2_user_settings VALUES ('admin_user_id', 'owner');
    ''')
    task = SimpleNamespace(user='owner', message_id='a' * 32, goal='in five minutes', prior_id='b' * 32)
    monkeypatch.setattr(proposals, 'load_bound_owner_exchange', lambda conn, task:
        SimpleNamespace(prior_message_id=task.prior_id, sha256=hashlib.sha256(task.goal.encode()).hexdigest()))
    conn.execute('BEGIN')
    now = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
    yield conn, task, now
    conn.rollback()
    conn.close()


def prepared(setup):
    conn, task, now = setup
    proposal = proposals.prepare_locked(conn, task, message='call the dentist',
        due_at=(now + timedelta(minutes=5)).isoformat(), timezone_name='UTC', now=now)
    return conn, task, now, proposal


def test_proposal_does_not_authorize_effect_or_expose_replacement(setup):
    conn, task, now, proposal = prepared(setup)
    with pytest.raises(proposals.ProposalUnavailable):
        proposals.consume_locked(conn, task, proposal_id=proposal['id'], now=now, is_cancelled=lambda: False)
    with pytest.raises(proposals.ProposalUnavailable, match='already_exists'):
        proposals.prepare_locked(conn, task, message='different content',
            due_at=(now + timedelta(minutes=5)).isoformat(), timezone_name='UTC', now=now)


def test_changed_reply_and_expiry_invalidate_confirmation(setup):
    conn, task, now, proposal = prepared(setup)
    task.goal = 'not five minutes; tomorrow'
    with pytest.raises(proposals.ProposalUnavailable, match='context_changed'):
        proposals.confirm_locked(conn, task, proposal_id=proposal['id'],
            proposal_sha256=proposal['proposal_sha256'], authenticated_owner='owner', now=now)
    task.goal = 'in five minutes'
    with pytest.raises(proposals.ProposalUnavailable, match='expired'):
        proposals.confirm_locked(conn, task, proposal_id=proposal['id'],
            proposal_sha256=proposal['proposal_sha256'], authenticated_owner='owner', now=now + timedelta(minutes=10))


@pytest.mark.parametrize('due_at,zone', [
    ('2026-11-20T12:05:00', 'UTC'),
    ('2026-11-20T12:05:00+01:00', 'UTC'),
    ('2026-11-20T11:59:59+00:00', 'UTC'),
    ('2026-11-20T12:05:00+00:00', 'Not/AZone'),
])
def test_invalid_or_unverified_time_is_rejected(setup, due_at, zone):
    conn, task, now = setup
    with pytest.raises(proposals.ProposalUnavailable, match='time_invalid'):
        proposals.prepare_locked(conn, task, message='dentist', due_at=due_at, timezone_name=zone, now=now)


def test_cancellation_after_insert_rolls_back_both_effect_and_consumption(setup):
    conn, task, now, proposal = prepared(setup)
    proposals.confirm_locked(conn, task, proposal_id=proposal['id'], proposal_sha256=proposal['proposal_sha256'],
                             authenticated_owner='owner', now=now)
    conn.execute('SAVEPOINT effect')
    checks = iter([False, True])
    with pytest.raises(proposals.ProposalUnavailable):
        proposals.consume_locked(conn, task, proposal_id=proposal['id'], now=now,
                                 is_cancelled=lambda: next(checks))
    conn.execute('ROLLBACK TO effect')
    conn.execute('RELEASE effect')
    assert conn.execute("SELECT state FROM owner_reminder_proposals WHERE id=?", (proposal['id'],)).fetchone()[0] == 'confirmed'
    # CREATE TABLE may also be rolled back by the savepoint.
    table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='owner_desktop_reminders'").fetchone()
    assert table is None or conn.execute('SELECT COUNT(*) FROM owner_desktop_reminders').fetchone()[0] == 0


def test_unrelated_existing_effect_is_not_claimed_as_new(setup):
    conn, task, now, proposal = prepared(setup)
    proposals.confirm_locked(conn, task, proposal_id=proposal['id'], proposal_sha256=proposal['proposal_sha256'],
                             authenticated_owner='owner', now=now)
    proposals.OwnerDesktopReminders.create_locked(conn, task.user, 'call the dentist', proposal['due_at'],
                                                  task.message_id, now=now)
    with pytest.raises(proposals.ProposalUnavailable, match='effect_unavailable'):
        proposals.consume_locked(conn, task, proposal_id=proposal['id'], now=now, is_cancelled=lambda: False)
    assert conn.execute('SELECT COUNT(*) FROM owner_desktop_reminders').fetchone()[0] == 1


def test_independent_original_requests_are_not_deduplicated(setup):
    conn, first_task, now, first = prepared(setup)
    proposals.confirm_locked(conn, first_task, proposal_id=first['id'], proposal_sha256=first['proposal_sha256'],
                             authenticated_owner='owner', now=now)
    proposals.consume_locked(conn, first_task, proposal_id=first['id'], now=now, is_cancelled=lambda: False)
    second_task = SimpleNamespace(user='owner', message_id='c' * 32, goal='in five minutes', prior_id='d' * 32)
    second = proposals.prepare_locked(conn, second_task, message='water the plants',
        due_at=(now + timedelta(minutes=5)).isoformat(), timezone_name='UTC', now=now)
    proposals.confirm_locked(conn, second_task, proposal_id=second['id'], proposal_sha256=second['proposal_sha256'],
                             authenticated_owner='owner', now=now)
    result = proposals.consume_locked(conn, second_task, proposal_id=second['id'], now=now, is_cancelled=lambda: False)
    assert result['status'] == 'created'
    assert conn.execute('SELECT COUNT(*) FROM owner_desktop_reminders').fetchone()[0] == 2


def test_cancelled_proposal_does_not_consume_original_intent(setup):
    conn, first_task, now, first = prepared(setup)
    proposals.cancel_locked(conn, first_task, proposal_id=first['id'], authenticated_owner='owner', now=now)
    next_reply = SimpleNamespace(user='owner', message_id='c' * 32, goal='ten minutes instead',
                                 prior_id=first_task.prior_id)
    next_proposal = proposals.prepare_locked(conn, next_reply, message='call the dentist',
        due_at=(now + timedelta(minutes=10)).isoformat(), timezone_name='UTC', now=now)
    proposals.confirm_locked(conn, next_reply, proposal_id=next_proposal['id'],
        proposal_sha256=next_proposal['proposal_sha256'], authenticated_owner='owner', now=now)
    result = proposals.consume_locked(conn, next_reply, proposal_id=next_proposal['id'], now=now,
                                      is_cancelled=lambda: False)
    assert result['status'] == 'created'
    assert conn.execute('SELECT COUNT(*) FROM owner_desktop_reminders').fetchone()[0] == 1


def test_no_work_outside_a_transaction(setup):
    conn, task, now = setup
    conn.commit()
    with pytest.raises(proposals.ProposalUnavailable, match='transaction_required'):
        proposals.prepare_locked(conn, task, message='dentist',
            due_at=(now + timedelta(minutes=5)).isoformat(), timezone_name='UTC', now=now)
