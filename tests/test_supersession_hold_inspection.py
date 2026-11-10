"""Owner-scoped hold inspection only; no household data or recovery mutation."""
from datetime import datetime, timezone
import json
import sqlite3

import pytest

from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.messages.supersession_hold import install_hold_schema, inspect_owned_hold, park_superseded
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def held(tmp_path):
    path = tmp_path / 'runtime.sqlite'
    users = V2UserStore(path)
    users.set_users_root(tmp_path / 'profiles')
    users.create_user(user_id='alice', display_name='Alice')
    users.create_user(user_id='bob', display_name='Bob')
    queue = SQLiteMessageQueue(path, lease_owner='fixture', experimental_claim_fencing=True)
    questions = SQLiteQuestionStore(path)
    now = datetime.now(timezone.utc)
    queue.enqueue(CoreMessage(timestamp=now, prompt='private request', user='alice'), message_id='message')
    receipt = queue.claim_next().claim_receipt
    task = TaskState(task_id='task', message_id='message', user='alice', goal='private request',
                     intelligence_engine='hierarchical_v3', intelligence_schema_version=3)
    version = questions.save_task_checkpoint(task, status='running')
    task.metadata['private_evidence'] = 'must-not-leak'
    questions.save_task_checkpoint(task, status='running')
    with sqlite3.connect(path) as conn:
        install_hold_schema(conn)
        assert park_superseded(conn, receipt, task_id='task', actor='alice', engine='hierarchical_v3',
                               observed_version=version, clock=lambda: now).status == 'held'
    return path, users, queue, questions, receipt


def inspect(path, actor='alice', message_id='message'):
    with sqlite3.connect(path) as conn:
        return inspect_owned_hold(conn, AccountSession(actor), message_id=message_id)


def test_active_owner_sees_only_safe_status_without_mutation(held):
    path, _, _, _, receipt = held
    with sqlite3.connect(path) as conn:
        before = [conn.execute('SELECT * FROM v2_supersession_holds').fetchall(),
                  conn.execute('SELECT task_state_json FROM v2_task_checkpoints').fetchall(),
                  conn.execute('SELECT * FROM v2_inbound_messages').fetchall()]
    result = inspect(path)
    assert (result.status, result.checkpoint_status, result.next_step) == (
        'held', 'running', 'requires_authorized_reconciliation')
    assert receipt.claim_generation not in repr(result) and 'private' not in repr(result)
    with sqlite3.connect(path) as conn:
        after = [conn.execute('SELECT * FROM v2_supersession_holds').fetchall(),
                 conn.execute('SELECT task_state_json FROM v2_task_checkpoints').fetchall(),
                 conn.execute('SELECT * FROM v2_inbound_messages').fetchall()]
    assert before == after


def test_other_or_inactive_actor_cannot_inspect_or_distinguish_absence(held):
    path, users, _, _, _ = held
    assert inspect(path, actor='bob') == inspect(path, actor='bob', message_id='unknown')
    assert inspect(path, actor='bob').status == 'unavailable'
    users.update_user('alice', is_active=False)
    assert inspect(path).status == 'unavailable'


def test_cancelled_task_remains_held_and_does_not_authorize_retry(held):
    path, _, _, questions, _ = held
    assert questions.mark_task_checkpoint_terminal('task', status='cancelled')
    result = inspect(path)
    assert result.status == 'held' and result.checkpoint_status == 'cancelled'
    assert result.next_step == 'requires_authorized_reconciliation'


@pytest.mark.parametrize('corruption', ['generation', 'checkpoint_owner', 'task', 'schema'])
def test_inconsistent_or_partial_deletion_stays_unavailable(held, corruption):
    path, _, _, _, _ = held
    with sqlite3.connect(path) as conn:
        if corruption == 'generation':
            conn.execute("UPDATE v2_supersession_holds SET claim_generation=? WHERE message_id='message'", ('f' * 32,))
        elif corruption == 'checkpoint_owner':
            conn.execute("UPDATE v2_task_checkpoints SET owner_id='bob' WHERE task_id='task'")
        elif corruption == 'task':
            conn.execute("DELETE FROM v2_task_checkpoints WHERE task_id='task'")
        else:
            row = conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()
            task = json.loads(row[0]); task['intelligence_schema_version'] = 999
            conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(task),))
    assert inspect(path).status == 'unavailable'


def test_missing_or_tampered_hold_schema_fails_closed(held):
    path, _, _, _, _ = held
    with sqlite3.connect(path) as conn:
        conn.execute('DROP TRIGGER v2_supersession_hold_prevent_delete')
    with pytest.raises(ClaimFenceUnavailable, match='hold_schema_unavailable'):
        inspect(path)


def test_explicit_trusted_session_required(held):
    path, _, _, _, _ = held
    with sqlite3.connect(path) as conn:
        with pytest.raises(ValueError, match='arguments_invalid'):
            inspect_owned_hold(conn, {'actor_user_id': 'alice'}, message_id='message')


def test_malformed_checkpoint_json_is_opaque(held):
    path, _, _, _, _ = held
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json='not-json' WHERE task_id='task'")
    assert inspect(path).status == 'unavailable'


@pytest.mark.parametrize('table', ['v2_users', 'v2_task_checkpoints', 'v2_inbound_messages'])
def test_missing_core_table_fails_closed(held, table):
    path, _, _, _, _ = held
    with sqlite3.connect(path) as conn:
        conn.execute(f'DROP TABLE {table}')
    with pytest.raises(ClaimFenceUnavailable):
        inspect(path)


@pytest.mark.parametrize('change', ['deactivate', 'cancel_checkpoint'])
def test_concurrent_authority_or_checkpoint_change_waits_for_inspection_snapshot(held, change):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    path, _, _, _, _ = held
    inspected_user, writer_attempted = Event(), Event()
    class Intercept(sqlite3.Connection):
        def execute(self, sql, parameters=(), /):
            result = super().execute(sql, parameters)
            if sql.startswith('SELECT 1 FROM main.v2_users'):
                inspected_user.set()
                assert writer_attempted.wait(timeout=5)
            return result
    def reader():
        with sqlite3.connect(path, factory=Intercept, timeout=5) as conn:
            return inspect_owned_hold(conn, AccountSession('alice'), message_id='message')
    def writer():
        assert inspected_user.wait(timeout=5)
        with sqlite3.connect(path, timeout=5) as conn:
            writer_attempted.set()
            if change == 'deactivate':
                conn.execute("UPDATE v2_users SET is_active=0 WHERE user_id='alice'")
            else:
                conn.execute("UPDATE v2_task_checkpoints SET status='cancelled',version=version+1 WHERE task_id='task'")
    with ThreadPoolExecutor(max_workers=2) as pool:
        reading = pool.submit(reader)
        writing = pool.submit(writer)
        before = reading.result(timeout=10)
        writing.result(timeout=10)
    assert before.status == 'held' and before.checkpoint_status == 'running'
    after = inspect(path)
    if change == 'deactivate':
        assert after.status == 'unavailable'
    else:
        assert after.status == 'held' and after.checkpoint_status == 'cancelled'
