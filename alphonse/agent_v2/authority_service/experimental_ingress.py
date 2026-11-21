"""Experimental two-store owner ingress; not a Desktop authentication boundary.

The client passed here must run in an authenticated Desktop context and speak
to a protected authority service. The existing same-user owner IPC is not
such a boundary. This code is deliberately opt-in and is not wired into the
normal Desktop, daemon configuration or host-Bash activation.
"""
from __future__ import annotations

import re


class ExperimentalIngressUnavailable(ValueError):
    pass


def submit_owner_request(authority_desktop, daemon_client, *, prompt, message_id,
                         reply_to_message_id=None):
    """Submit authority first; enqueue only after receiving its request ID.

    The caller must durably retain the message ID before invoking this helper,
    so a lost response can be inspected. This does not automatically retry,
    cancel or reconcile an orphan in the separate service store.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ExperimentalIngressUnavailable('owner_authority_ingress_invalid')
    if not isinstance(message_id, str) or re.fullmatch(r'[0-9a-f]{32}', message_id) is None:
        raise ExperimentalIngressUnavailable('owner_authority_ingress_invalid')
    goal = prompt.strip()
    try:
        issued = authority_desktop.submit(goal=goal, idempotency_key=message_id)
        if (not isinstance(issued, dict) or set(issued) != {'request_id'}
                or not isinstance(issued['request_id'], str)
                or re.fullmatch(r'[0-9a-f]{32}', issued['request_id']) is None):
            raise ValueError()
        params = {'prompt': goal, 'message_id': message_id, 'authority_request_id': issued['request_id']}
        if reply_to_message_id is not None:
            params['reply_to_message_id'] = reply_to_message_id
        receipt = daemon_client.request('experimental_owner_ingress', **params)
        if not isinstance(receipt, dict) or receipt.get('message_id') != message_id:
            raise ValueError()
        return receipt
    except Exception:
        # An authority record or queue row might exist after a lost response.
        # The caller must inspect/reconcile before another attempt; this helper
        # never fabricates a receipt or repeats an uncertain mutation.
        raise ExperimentalIngressUnavailable('owner_authority_ingress_unconfirmed') from None


def queue_request_binding(queue, task):
    """Read a local lookup hint; service claim must independently verify it."""
    from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
    if (type(queue) is not SQLiteMessageQueue or not isinstance(getattr(task, 'task_id', None), str)
            or re.fullmatch(r'[0-9a-f]{32}', task.task_id) is None
            or getattr(task, 'message_id', None) != task.task_id):
        raise ExperimentalIngressUnavailable('owner_authority_binding_unavailable')
    try:
        with queue._connect() as conn:
            row = conn.execute('SELECT request_id FROM v2_authority_ingress_bindings WHERE message_id=?',
                               (task.task_id,)).fetchone()
        if row is None or re.fullmatch(r'[0-9a-f]{32}', row['request_id']) is None:
            raise ValueError()
        return row['request_id']
    except Exception:
        raise ExperimentalIngressUnavailable('owner_authority_binding_unavailable') from None
