"""Claim-bound bootstrap for disposable direct-owner candidate experiments.

Not an authenticated entry point or a production engine. Legacy writers, user
revocation outside this database, all conversation sinks, daemon handoff and
claim recovery remain unfenced. Never use this on a household database.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
import sqlite3
from typing import Callable

from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import (
    Checkpoint, CheckpointConflict, ENGINE, SCHEMA_VERSION, STATE_KEY, ExperimentalCheckpointStore,
)
from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits, TaskBudget
from .claim_fence import ClaimFenceUnavailable, ClaimReceipt, _time, fenced_claim_transaction


def bootstrap_fixture_task(store: ExperimentalCheckpointStore, claim: ClaimReceipt, *,
                           actor: str, limits: BudgetLimits,
                           clock: Callable[[], datetime]) -> Checkpoint:
    """Create one candidate and budget from the live claimed row, atomically.

    The test host must authenticate the actor separately. Only new, plain,
    direct-owner messages in one disposable SQLite file are supported. Metadata,
    projects, continuations, preexisting checkpoints and cross-store work are
    rejected rather than implicitly interpreted. The goal source is a host
    reference, not permission to read or disclose the message.
    """
    if (not isinstance(store, ExperimentalCheckpointStore) or not isinstance(claim, ClaimReceipt)
            or not isinstance(actor, str) or not actor.strip() or actor != claim.user_id
            or not isinstance(limits, BudgetLimits) or not callable(clock)):
        raise ValueError('fixture_bootstrap_arguments_invalid')
    previous = None
    def sample():
        nonlocal previous
        try:
            now = _time(clock())
        except Exception:
            raise ClaimFenceUnavailable('fixture_bootstrap_clock_unavailable') from None
        if previous is not None and now < previous:
            raise ClaimFenceUnavailable('fixture_bootstrap_clock_reversed')
        previous = now
        return now

    try:
        with store.store._connect() as conn:
            path = next((row[2] for row in conn.execute('PRAGMA database_list') if row[1] == 'main'), '')
            if not path or not os.path.samefile(path, store.store.db_path):
                raise ClaimFenceUnavailable('fixture_bootstrap_store_unavailable')
            with fenced_claim_transaction(conn, claim, clock=sample):
                row = conn.execute('SELECT * FROM main.v2_inbound_messages WHERE message_id=?',
                                   (claim.message_id,)).fetchone()
                if (row is None or conn.execute(
                        'SELECT 1 FROM main.v2_users WHERE user_id=? AND is_active=1', (actor,)).fetchone() is None):
                    raise ClaimFenceUnavailable('fixture_bootstrap_unavailable')
                try:
                    metadata = json.loads(row['metadata_json'])
                except (TypeError, ValueError):
                    raise ClaimFenceUnavailable('fixture_bootstrap_unavailable') from None
                prompt = row['prompt']
                if (metadata != {} or not isinstance(prompt, str) or not prompt.strip()
                        or len(prompt.encode('utf-8')) > 12000
                        or any(row[key] for key in ('project_id', 'tag', 'correlation_id', 'memory_session_id'))):
                    raise ClaimFenceUnavailable('fixture_bootstrap_unsupported_message')
                if conn.execute('SELECT 1 FROM main.v2_scheduled_handoffs WHERE message_id=?',
                                (claim.message_id,)).fetchone() is not None:
                    raise ClaimFenceUnavailable('fixture_bootstrap_unsupported_message')
                # A different task ID may already refer to this message. Do not
                # guess which is authoritative; malformed rows are opaque too.
                if conn.execute('''SELECT 1 FROM main.v2_task_checkpoints
                    WHERE task_id=? OR CASE WHEN json_valid(task_state_json)
                        AND json_type(task_state_json)='object'
                        THEN json_extract(task_state_json, '$.message_id')=? ELSE 1 END LIMIT 1''',
                        (claim.message_id, claim.message_id)).fetchone() is not None:
                    raise CheckpointConflict('fixture_bootstrap_checkpoint_exists')
                started = sample()
                state = CandidateState(claim.message_id, actor, 'queued_goal')
                budget = TaskBudget(limits, started_at_ms=int(started.timestamp() * 1000))
                task = TaskState(task_id=claim.message_id, message_id=claim.message_id, user=actor,
                                 goal=prompt, intelligence_engine=ENGINE, intelligence_schema_version=1)
                raw = task.to_checkpoint_dict()
                raw['metadata'][STATE_KEY] = {'schema_version': SCHEMA_VERSION,
                                              'state': state.snapshot(), 'budget': budget.to_dict()}
                stamp = started.isoformat()
                encoded = json.dumps(raw, sort_keys=True)
                conn.execute('''INSERT INTO main.v2_task_checkpoints
                    (task_id,task_state_json,status,owner_id,project_id,correlation_id,version,created_at,updated_at)
                    VALUES (?,?,'running',?,'','',1,?,?)''',
                    (claim.message_id, encoded, actor, stamp, stamp))
                saved = store._row(conn, claim.message_id)
                checked = store._validate_identity(saved, claim.message_id, actor, 1)
                restored_state, restored_budget = store._decode(checked, claim.message_id, actor)
                full = conn.execute('SELECT * FROM main.v2_task_checkpoints WHERE task_id=?',
                                    (claim.message_id,)).fetchone()
                expected = {'task_id': claim.message_id, 'task_state_json': encoded, 'status': 'running',
                            'owner_id': actor, 'project_id': '', 'correlation_id': '', 'version': 1,
                            'created_at': stamp, 'updated_at': stamp}
                if (checked != raw or restored_state.snapshot() != state.snapshot()
                        or restored_budget.to_dict() != budget.to_dict()
                        or full is None or dict(full) != expected
                        or conn.execute('''SELECT count(*) FROM main.v2_task_checkpoints
                            WHERE CASE WHEN json_valid(task_state_json) AND json_type(task_state_json)='object'
                                THEN json_extract(task_state_json, '$.message_id')=? ELSE 1 END''',
                                        (claim.message_id,)).fetchone()[0] != 1
                        or conn.execute('SELECT 1 FROM main.v2_users WHERE user_id=? AND is_active=1',
                                        (actor,)).fetchone() is None):
                    raise ClaimFenceUnavailable('fixture_bootstrap_changed')
                return Checkpoint(1, state, budget)
    except (OSError, sqlite3.Error, StopIteration, UnicodeError):
        raise ClaimFenceUnavailable('fixture_bootstrap_storage_unavailable') from None
