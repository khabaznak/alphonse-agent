"""Owner mode must not accept work it cannot acquire after a mode switch."""
from types import SimpleNamespace
from datetime import datetime, timezone
import sqlite3

from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue


def test_owner_readiness_fails_closed_when_queue_is_unavailable():
    def unavailable():
        raise OSError('fixture database unavailable')

    daemon = SimpleNamespace(experimental_owner_only=True,
        _processor_thread=SimpleNamespace(is_alive=lambda: True),
        _stop=SimpleNamespace(is_set=lambda: False),
        runtime=SimpleNamespace(queue=SimpleNamespace(claim_mode_blocked=unavailable)))
    assert V2Daemon.owner_processing_blocking_reason(daemon) == 'queue_unavailable'
    daemon.owner_processing_blocking_reason = lambda: V2Daemon.owner_processing_blocking_reason(daemon)
    assert V2Daemon.owner_processing_ready(daemon) is False
    assert V2Daemon.owner_paused_legacy_requests(daemon) is None


def test_legacy_failed_messages_do_not_block_fenced_owner_worker(tmp_path):
    path = tmp_path / 'legacy.sqlite'
    old = SQLiteMessageQueue(path)
    for number in range(50):
        old.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt='synthetic', user='owner'),
                    message_id=f'legacy-{number}')
    with sqlite3.connect(path) as db:
        assert 'claim_generation' not in {row[1] for row in db.execute('PRAGMA table_info(v2_inbound_messages)')}
        db.execute("UPDATE v2_inbound_messages SET status='failed'")
    current = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    assert current.claim_mode_blocked() is False
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM v2_inbound_messages WHERE status='failed' AND claim_generation=''").fetchone()[0] == 50
