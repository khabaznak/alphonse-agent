"""Opt-in holds use disposable SQLite; no runtime activation or live data."""
from datetime import datetime, timedelta, timezone
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimReceipt, ClaimSuperseded, new_claim_generation
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.messages.supersession_hold import install_hold_schema, park_superseded
from alphonse.agent_v2.core.questions import SQLiteQuestionStore


@pytest.fixture
def f(tmp_path):
    path = tmp_path / 'runtime.sqlite3'
    queue = SQLiteMessageQueue(path, lease_owner='worker')
    questions = SQLiteQuestionStore(path)
    now = datetime.now(timezone.utc)
    queue.enqueue(CoreMessage(timestamp=now, prompt='synthetic', user='alice'), message_id='message')
    queue.claim_next(lease_owner='worker', lease_seconds=60)
    task = TaskState(task_id='task', message_id='message', user='alice', goal='synthetic',
                     intelligence_engine='hierarchical_v3', intelligence_schema_version=3)
    version = questions.save_task_checkpoint(task, status='running')
    generation = new_claim_generation()
    with queue._connect() as conn:
        conn.execute("ALTER TABLE v2_inbound_messages ADD COLUMN claim_generation TEXT NOT NULL DEFAULT ''")
        conn.execute('UPDATE v2_inbound_messages SET claim_generation=? WHERE message_id=?', (generation, 'message'))
    with sqlite3.connect(path) as conn:
        install_hold_schema(conn)
    claim = ClaimReceipt('message', 'alice', 'worker', generation)
    return path, queue, questions, claim, version, now


def park(f, *, actor='alice', task_id='task', engine='hierarchical_v3', version=None, clock=None, claim=None):
    path, _, _, receipt, observed, now = f
    with sqlite3.connect(path) as conn:
        return park_superseded(conn, claim or receipt, task_id=task_id, actor=actor, engine=engine,
            observed_version=observed if version is None else version, clock=clock or (lambda: now))


def advance(f, status='running'):
    task = f[2].load_task_checkpoint('task')
    task.metadata['unconfirmed_effect'] = {'synthetic': 'unknown'}
    f[2].save_task_checkpoint(task, status=status)


def test_opt_in_schema_idempotent_and_absent_by_default(tmp_path, f):
    untouched = tmp_path / 'other.sqlite3'
    q = SQLiteMessageQueue(untouched)
    with q._connect() as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='v2_supersession_holds'").fetchone() is None
    with sqlite3.connect(f[0]) as conn:
        install_hold_schema(conn)


def test_hold_preserves_checkpoint_claim_and_blocks_legacy_ack_retry_and_delete(f):
    advance(f)
    assert park(f).status == 'held'
    assert park(f).status == 'already_held'
    with sqlite3.connect(f[0]) as conn:
        held = conn.execute('SELECT task_id,checkpoint_status FROM v2_supersession_holds').fetchone()
        assert held == ('task', 'running')
        assert conn.execute('SELECT status FROM v2_inbound_messages').fetchone()[0] == 'processing'
        assert conn.execute('SELECT status FROM v2_task_checkpoints').fetchone()[0] == 'running'
        with pytest.raises(sqlite3.IntegrityError, match='requires_reconciliation'):
            conn.execute("DELETE FROM v2_inbound_messages WHERE message_id='message'")
        with pytest.raises(sqlite3.IntegrityError, match='requires_reconciliation'):
            conn.execute("UPDATE v2_inbound_messages SET status='retry_wait' WHERE message_id='message'")
        # SQLite's REPLACE conflict handling does not fire DELETE triggers by
        # default. A dedicated BEFORE INSERT guard must stop this path too.
        assert conn.execute('PRAGMA recursive_triggers').fetchone()[0] == 0
        with pytest.raises(sqlite3.IntegrityError, match='requires_reconciliation'):
            conn.execute('''INSERT OR REPLACE INTO v2_inbound_messages
                (message_id,sequence,timestamp,prompt,user_id,queued_at,status)
                VALUES ('message',1,'2026-01-01T00:00:00+00:00','replacement','alice',
                        '2026-01-01T00:00:00+00:00','pending')''')
    assert f[2].load_task_checkpoint('task').metadata['unconfirmed_effect'] == {'synthetic': 'unknown'}


@pytest.mark.parametrize('guard', ['insert', 'update', 'delete'])
@pytest.mark.parametrize('already_held', [False, True])
@pytest.mark.parametrize('replace', [False, True])
def test_park_refuses_missing_or_replaced_guard_even_for_existing_hold(f, guard, already_held, replace):
    advance(f)
    if already_held:
        assert park(f).status == 'held'
    name = f'v2_supersession_hold_prevent_{guard}'
    with sqlite3.connect(f[0]) as conn:
        conn.execute(f'DROP TRIGGER {name}')
        if replace:
            # A same-named but ineffective trigger must not be adopted either.
            conn.execute(f'CREATE TRIGGER {name} BEFORE {guard.upper()} ON v2_inbound_messages BEGIN SELECT 1; END')
    with pytest.raises(ClaimFenceUnavailable, match='hold_schema_unavailable'):
        park(f)
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM v2_supersession_holds').fetchone()[0] == int(already_held)
        assert conn.execute('SELECT status FROM v2_inbound_messages').fetchone()[0] == 'processing'


def test_cancellation_without_version_increment_can_be_held(f):
    assert f[2].mark_task_checkpoint_terminal('task', status='cancelled')
    assert park(f).status == 'held'
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT checkpoint_status FROM v2_supersession_holds').fetchone()[0] == 'cancelled'
        assert conn.execute('SELECT status FROM v2_task_checkpoints').fetchone()[0] == 'cancelled'


def test_no_proven_supersession_and_mismatches_do_not_park(f):
    assert park(f).status == 'not_superseded'
    assert park(f, actor='bob').status == 'unavailable'
    assert park(f, task_id='missing').status == 'unavailable'
    assert park(f, engine='tactical_v2').status == 'unavailable'
    assert park(f, version=f[4] + 1).status == 'unavailable'
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM v2_supersession_holds').fetchone()[0] == 0


def test_read_only_deletion_preflight_can_race_a_new_hold(f):
    """Document why a preflight query alone cannot protect multi-store deletion."""
    advance(f)
    with sqlite3.connect(f[0]) as deleting:
        assert deleting.execute(
            'SELECT 1 FROM v2_supersession_holds WHERE user_id=?', ('alice',)
        ).fetchone() is None
    # A different worker can commit a hold after the query but before deletion.
    assert park(f).status == 'held'
    with sqlite3.connect(f[0]) as deleting:
        with pytest.raises(sqlite3.IntegrityError, match='requires_reconciliation'):
            deleting.execute('DELETE FROM v2_inbound_messages WHERE user_id=?', ('alice',))
    with sqlite3.connect(f[0]) as check:
        assert check.execute('SELECT count(*) FROM v2_supersession_holds').fetchone()[0] == 1
        assert check.execute('SELECT count(*) FROM v2_task_checkpoints').fetchone()[0] == 1


def test_stale_and_expired_claims_cannot_park(f):
    advance(f)
    with pytest.raises(ClaimSuperseded):
        park(f, claim=ClaimReceipt('message', 'alice', 'worker', new_claim_generation()))
    with pytest.raises(ClaimSuperseded):
        park(f, clock=lambda: f[5] + timedelta(days=1))
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM v2_supersession_holds').fetchone()[0] == 0


def test_expiry_during_transaction_rolls_back_hold(f):
    advance(f)
    moments = iter((f[5], f[5], f[5] + timedelta(days=1)))
    with pytest.raises(ClaimSuperseded):
        park(f, clock=lambda: next(moments))
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM v2_supersession_holds').fetchone()[0] == 0


def test_crash_after_hold_insert_before_commit_rolls_back(f):
    advance(f)
    calls = 0
    def interrupted():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt()
        return f[5]
    with pytest.raises(KeyboardInterrupt):
        park(f, clock=interrupted)
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM v2_supersession_holds').fetchone()[0] == 0
        assert conn.execute('SELECT status FROM v2_inbound_messages').fetchone()[0] == 'processing'


def test_hold_survives_reopen_and_prevents_same_owner_aba(f):
    advance(f)
    assert park(f).status == 'held'
    with sqlite3.connect(f[0]) as reopened:
        with pytest.raises(sqlite3.IntegrityError, match='requires_reconciliation'):
            reopened.execute("UPDATE v2_inbound_messages SET claim_generation=? WHERE message_id='message'", (new_claim_generation(),))
        assert reopened.execute('SELECT count(*) FROM v2_supersession_holds').fetchone()[0] == 1


def test_missing_or_malformed_checkpoint_remains_opaque(f):
    advance(f)
    with sqlite3.connect(f[0]) as conn:
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json='not-json' WHERE task_id='task'")
    assert park(f).status == 'unavailable'
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM v2_supersession_holds').fetchone()[0] == 0


def test_delegated_queue_actor_is_not_assumed_to_be_task_owner(f):
    advance(f)
    with sqlite3.connect(f[0]) as conn:
        conn.execute("UPDATE v2_inbound_messages SET user_id='bob' WHERE message_id='message'")
    delegated = ClaimReceipt('message', 'bob', 'worker', f[3].claim_generation)
    assert park(f, claim=delegated, actor='alice').status == 'unavailable'
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM v2_supersession_holds').fetchone()[0] == 0


def test_concurrent_newer_checkpoint_waits_for_atomic_hold_commit(f):
    advance(f)
    entered, competing = Event(), Event()
    count = 0
    def clock():
        nonlocal count
        count += 1
        if count == 1:
            entered.set()
            assert competing.wait(timeout=5)
        return f[5]
    def competing_writer():
        assert entered.wait(timeout=5)
        with sqlite3.connect(f[0], timeout=5) as conn:
            competing.set()
            conn.execute("UPDATE v2_task_checkpoints SET version=version+1 WHERE task_id='task'")
    with ThreadPoolExecutor(max_workers=2) as pool:
        parked = pool.submit(park, f, clock=clock)
        writer = pool.submit(competing_writer)
        assert parked.result(timeout=10).status == 'held'
        writer.result(timeout=10)
    with sqlite3.connect(f[0]) as conn:
        hold_version = conn.execute('SELECT current_version FROM v2_supersession_holds').fetchone()[0]
        latest = conn.execute('SELECT version FROM v2_task_checkpoints').fetchone()[0]
        assert latest == hold_version + 1
        assert conn.execute('SELECT status FROM v2_inbound_messages').fetchone()[0] == 'processing'


def test_missing_generation_or_conflicting_schema_refuses_install(tmp_path):
    path = tmp_path / 'missing.sqlite3'
    SQLiteMessageQueue(path)
    with sqlite3.connect(path) as conn:
        with pytest.raises(ClaimFenceUnavailable, match='prerequisites'):
            install_hold_schema(conn)
        conn.execute("ALTER TABLE v2_inbound_messages ADD COLUMN claim_generation TEXT NOT NULL DEFAULT ''")
        conn.execute('CREATE TABLE v2_supersession_holds (message_id TEXT)')
        conn.commit()
        with pytest.raises(ClaimFenceUnavailable, match='conflict'):
            install_hold_schema(conn)


def test_real_opt_in_queue_hold_survives_restart_and_does_not_block_other_reclaims(tmp_path):
    path = tmp_path / 'runtime.sqlite3'
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    questions = SQLiteQuestionStore(path)
    now = datetime.now(timezone.utc)
    queue.enqueue(CoreMessage(timestamp=now, prompt='synthetic', user='alice'), message_id='message')
    claimed = queue.claim_next(lease_owner='worker', lease_seconds=60)
    task = TaskState(task_id='task', message_id='message', user='alice', goal='synthetic',
                     intelligence_engine='hierarchical_v3', intelligence_schema_version=3)
    version = questions.save_task_checkpoint(task, status='running')
    task.metadata['effect_state'] = 'unknown'
    questions.save_task_checkpoint(task, status='running')
    with sqlite3.connect(path) as conn:
        install_hold_schema(conn)
        assert park_superseded(conn, claimed.claim_receipt, task_id='task', actor='alice',
            engine='hierarchical_v3', observed_version=version, clock=lambda: now).status == 'held'
    # Other claims can expire and be reclaimed without touching the held row.
    queue.enqueue(CoreMessage(timestamp=now, prompt='other', user='alice'), message_id='other')
    other = queue.claim_next(lease_owner='worker', lease_seconds=60)
    assert other.message_id == 'other'
    reopened = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    assert reopened.reclaim_expired(now=now + timedelta(days=1)) == 1
    assert reopened.ack_fenced(claimed.claim_receipt) is False
    assert reopened.ack('message', lease_owner='worker') is False
    assert reopened.retry('message', lease_owner='worker') is False
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT status FROM v2_inbound_messages WHERE message_id='message'").fetchone()[0] == 'processing'
        assert conn.execute("SELECT status FROM v2_inbound_messages WHERE message_id='other'").fetchone()[0] == 'pending'
        assert conn.execute('SELECT count(*) FROM v2_supersession_holds').fetchone()[0] == 1
    assert questions.load_task_checkpoint('task').metadata['effect_state'] == 'unknown'
