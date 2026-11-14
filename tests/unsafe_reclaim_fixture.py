"""Simulate an unsafe old writer for stale-claim tests only.

This direct SQL intentionally bypasses the fail-closed queue API. It is not a
recovery procedure or evidence that replaying the message is safe.
"""


def simulate_old_writer_requeue(queue, message_id):
    with queue._connect() as conn:
        changed = conn.execute("""UPDATE v2_inbound_messages SET status='pending',
            processing_at='',lease_owner='',lease_expires_at=''
            WHERE message_id=? AND status='processing'""", (message_id,))
        assert changed.rowcount == 1
