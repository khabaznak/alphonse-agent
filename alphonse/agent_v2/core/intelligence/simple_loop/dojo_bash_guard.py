"""An explicit, in-process Dojo capability; never a household shell grant."""
from __future__ import annotations

import re


def valid_dojo_session(context, session_id):
    """Require the exact live disposable executor and its private session nonce."""
    if not isinstance(session_id, str) or re.fullmatch(r'[0-9a-f]{32}', session_id) is None:
        return False
    from alphonse.agent_v2.evaluation.dojo_bash import DojoBash
    shell = getattr(context, '_dojo_bash_service', None)
    return (type(shell) is DojoBash and shell._active is True
            and getattr(context, '_dojo_bash_session', None) == session_id)


def valid_dojo_observation(value):
    """Check a bounded command observation; success is not effect proof."""
    if not isinstance(value, dict) or set(value) != {
            'candidate_complete', 'status', 'exit_code', 'stdout', 'stderr',
            'truncated', 'cleanup_verified', 'duration_ms'}:
        return False
    return (value['candidate_complete'] is True and value['status'] in {'completed', 'failed'}
            and type(value['exit_code']) is int and (value['exit_code'] == 0) == (value['status'] == 'completed')
            and isinstance(value['stdout'], str) and isinstance(value['stderr'], str)
            and len(value['stdout'].encode()) <= 65536 and len(value['stderr'].encode()) <= 65536
            and type(value['truncated']) is bool and value['cleanup_verified'] is None
            and type(value['duration_ms']) is int and value['duration_ms'] >= 0)
