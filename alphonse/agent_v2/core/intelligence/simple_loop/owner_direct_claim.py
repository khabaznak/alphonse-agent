"""Verify direct owner requests without confusing channel provenance."""
from __future__ import annotations


def verify_direct_owner_claim(queue, receipt, task):
    """Verify a current Desktop or mapped Telegram claim; schedules are excluded."""
    from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
    from .owner_telegram_ingress import verify_telegram_owner_claim
    if type(queue) is not SQLiteMessageQueue:
        return False
    try:
        return (queue.verify_trusted_local_owner_claim(receipt, task)
                or verify_telegram_owner_claim(queue, receipt, task))
    except Exception:
        return False


def verify_owner_execution_claim(queue, receipt, task):
    """Verify a direct owner request or a due, provenance-bound owner occurrence.

    This is a point-in-time identity and claim check. It does not grant any
    particular tool or account access; those checks remain tool-specific.
    """
    from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
    if type(queue) is not SQLiteMessageQueue:
        return False
    try:
        return (verify_direct_owner_claim(queue, receipt, task)
                or queue.verify_trusted_scheduled_owner_claim(receipt, task))
    except Exception:
        return False
