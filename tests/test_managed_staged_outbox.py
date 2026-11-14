"""Disposable staging checks; no provider call or production authority token."""
from datetime import timedelta
import sqlite3

import pytest

from test_single_loop_delivery_reservation import setup
from test_single_loop_managed_delivery import managed, invoke
from alphonse.agent_v2.core.io.experimental_staged_outbox import (
    install_experimental_staging, stage_managed_experimental_response,
    stage_experimental_response, restage_experimental_response_after_restart,
    restage_managed_experimental_response_after_restart,
)
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimSuperseded


@pytest.fixture
def ready(managed):
    base, bindings, session, _ = managed
    saved = invoke(managed)
    outbox = SQLiteOutboundStore(base[2].db_path)
    install_experimental_staging(outbox)
    return base, bindings, session, saved, outbox


def stage(f, **changes):
    base, bindings, session, saved, outbox = f
    args = dict(checkpoints=base[3], bindings=bindings, session=session, task_id='task',
                expected_version=saved.version, message='synthetic message', clock=lambda: base[6])
    args.update(changes)
    return stage_managed_experimental_response(outbox, base[4], **args)


def test_current_managed_authority_stages_only_inert_row(ready):
    base, _, _, _, outbox = ready
    identity = stage(ready)
    assert stage(ready) == identity
    assert outbox.claim_next() is None
    with sqlite3.connect(base[2].db_path) as conn:
        row = conn.execute('SELECT status,message,metadata_json FROM v2_outbox WHERE outbox_message_id=?', (identity,)).fetchone()
    assert row[0:2] == ('staged', 'synthetic message')
    assert 'fixture-secret' not in row[2]


def test_read_operation_cannot_stage_managed_delivery(ready):
    import json
    base, _, _, saved, _ = ready
    with sqlite3.connect(base[2].db_path) as conn:
        raw = json.loads(conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0])
        raw['metadata']['single_loop_managed_preflight']['operation'] = 'read'
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(raw),))
    with pytest.raises(ClaimFenceUnavailable):
        stage(ready)
    with sqlite3.connect(base[2].db_path) as conn:
        assert conn.execute('SELECT count(*) FROM v2_outbox').fetchone()[0] == 0
    assert base[3].load('task', 'alice', expected_version=saved.version).state.inflight == 'action'


def test_generic_entrypoints_cannot_bypass_managed_authority(ready):
    base, bindings, session, saved, outbox = ready
    action = base[2].load_task_checkpoint('task').metadata['single_loop_delivery_action']
    args = dict(checkpoints=base[3], task_id='task', actor='alice', expected_version=saved.version,
                integration_id=action['integration_id'], provider_key=action['provider_key'],
                channel_target=action['channel_target'], message='synthetic message',
                idempotency_key=action['idempotency_key'], clock=lambda: base[6])
    with pytest.raises(ClaimFenceUnavailable, match='managed_stage_authority_required'):
        stage_experimental_response(outbox, base[4], **args)
    identity = stage(ready)
    bindings.accounts.revoke_grant(session, grant_id='grant-1', expected_revision=1)
    with pytest.raises(ClaimFenceUnavailable, match='managed_stage_authority_required'):
        restage_experimental_response_after_restart(base[2].db_path, base[4], **args)
    with sqlite3.connect(base[2].db_path) as conn:
        assert conn.execute('SELECT status FROM v2_outbox WHERE outbox_message_id=?', (identity,)).fetchone()[0] == 'staged'


def test_removed_managed_preflight_cannot_fall_back_to_synthetic_staging(ready):
    """Missing managed metadata cannot change the action's authority mode."""
    import json
    base, _, _, saved, outbox = ready
    with sqlite3.connect(base[2].db_path) as conn:
        raw = conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0]
        task = json.loads(raw)
        action = task['metadata']['single_loop_delivery_action']
        assert action['authority_mode'] == 'managed'
        del task['metadata']['single_loop_managed_preflight']
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(task),))
    args = dict(checkpoints=base[3], task_id='task', actor='alice', expected_version=saved.version,
                integration_id=action['integration_id'], provider_key=action['provider_key'],
                channel_target=action['channel_target'], message='synthetic message',
                idempotency_key=action['idempotency_key'], clock=lambda: base[6])
    with pytest.raises(ClaimFenceUnavailable, match='managed_stage_authority_required'):
        stage_experimental_response(outbox, base[4], **args)
    with sqlite3.connect(base[2].db_path) as conn:
        assert conn.execute('SELECT count(*) FROM v2_outbox').fetchone()[0] == 0
    assert base[3].load('task', 'alice', expected_version=saved.version).state.inflight == 'action'


def test_managed_restart_requires_existing_row_and_current_access(ready):
    base, bindings, session, saved, _ = ready
    args = dict(checkpoints=base[3], bindings=bindings, session=session, task_id='task',
                expected_version=saved.version, message='synthetic message', clock=lambda: base[6])
    with pytest.raises(ClaimFenceUnavailable, match='reconciliation_required'):
        restage_managed_experimental_response_after_restart(base[2].db_path, base[4], **args)
    identity = stage(ready)
    assert restage_managed_experimental_response_after_restart(base[2].db_path, base[4], **args) == identity
    bindings.accounts.revoke_grant(session, grant_id='grant-1', expected_revision=1)
    with pytest.raises(ClaimFenceUnavailable):
        restage_managed_experimental_response_after_restart(base[2].db_path, base[4], **args)
    with sqlite3.connect(base[2].db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM v2_outbox').fetchone()[0] == 1


def test_grant_revoked_after_resolution_cannot_stage(ready, monkeypatch):
    base, bindings, session, saved, _ = ready
    original = bindings.resolve_account
    def resolve(*args, **kwargs):
        result = original(*args, **kwargs)
        bindings.accounts.revoke_grant(session, grant_id='grant-1', expected_revision=1)
        return result
    monkeypatch.setattr(bindings, 'resolve_account', resolve)
    with pytest.raises(ClaimFenceUnavailable):
        stage(ready)
    with sqlite3.connect(base[2].db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM v2_outbox').fetchone()[0] == 0
    assert base[3].load('task', 'alice', expected_version=saved.version).state.inflight == 'action'


@pytest.mark.parametrize('changed', ['binding', 'choice', 'actor', 'vault', 'skill', 'content'])
def test_changed_authority_or_content_never_stages(ready, monkeypatch, changed):
    base, bindings, _, saved, _ = ready
    if changed in {'binding', 'choice', 'actor'}:
        with bindings.integrations._connect() as conn:
            if changed == 'binding':
                conn.execute("UPDATE v2_skill_bindings SET enabled=0 WHERE binding_id='binding-one'")
            elif changed == 'choice':
                conn.execute("DELETE FROM v2_skill_binding_choices WHERE actor_id='alice'")
            else:
                conn.execute("UPDATE v2_users SET is_active=0 WHERE user_id='alice'")
    elif changed == 'vault':
        monkeypatch.setattr(bindings.integrations.credential_vault, 'health', lambda *args: 'unavailable')
    elif changed == 'skill':
        monkeypatch.setattr(bindings.skills, 'can_use', lambda *args: False)
    with pytest.raises(ClaimFenceUnavailable):
        stage(ready, **({'message': 'different'} if changed == 'content' else {}))
    with sqlite3.connect(base[2].db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM v2_outbox').fetchone()[0] == 0
    assert base[3].load('task', 'alice', expected_version=saved.version).state.inflight == 'action'


def test_claim_expires_between_resolution_and_staging(ready):
    base, _, _, saved, _ = ready
    samples = iter((base[6], base[6], base[6] + timedelta(days=1)))
    with pytest.raises(ClaimSuperseded):
        stage(ready, clock=lambda: next(samples))
    with sqlite3.connect(base[2].db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM v2_outbox').fetchone()[0] == 0
    assert base[3].load('task', 'alice', expected_version=saved.version).state.inflight == 'action'


def test_clock_reversal_between_checks_does_not_stage(ready):
    base = ready[0]
    samples = iter((base[6], base[6], base[6] - timedelta(milliseconds=1)))
    with pytest.raises(ClaimFenceUnavailable, match='clock_reversed'):
        stage(ready, clock=lambda: next(samples))
    with sqlite3.connect(base[2].db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM v2_outbox').fetchone()[0] == 0


def test_revocation_cannot_write_during_final_stage_transaction(ready, monkeypatch):
    from alphonse.agent_v2.core.io import experimental_staged_outbox as module
    base = ready[0]
    capture = module._capture
    calls = []
    def wrapped(conn, **kwargs):
        value = capture(conn, **kwargs)
        calls.append(1)
        if len(calls) == 2:
            with sqlite3.connect(base[2].db_path, timeout=0) as other:
                with pytest.raises(sqlite3.OperationalError, match='locked'):
                    other.execute("UPDATE v2_account_grants SET revision=revision+1 WHERE grant_id='grant-1'")
        return value
    monkeypatch.setattr(module, '_capture', wrapped)
    assert stage(ready)
    assert len(calls) == 2


def test_existing_stage_does_not_bypass_later_revocation(ready):
    base, bindings, session, _, _ = ready
    identity = stage(ready)
    bindings.accounts.revoke_grant(session, grant_id='grant-1', expected_revision=1)
    with pytest.raises(ClaimFenceUnavailable):
        stage(ready)
    with sqlite3.connect(base[2].db_path) as conn:
        assert conn.execute('SELECT status FROM v2_outbox WHERE outbox_message_id=?', (identity,)).fetchone()[0] == 'staged'
