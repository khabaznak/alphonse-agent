"""Disposable stage → fixture dispatch → fixture settlement, not production delivery."""
from datetime import timedelta
from unsafe_reclaim_fixture import simulate_old_writer_requeue
import json
import sqlite3

import pytest

from test_experimental_staged_outbox import setup, stage
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.v3.single_loop_delivery_reservation import DeliveryBinding
from alphonse.agent_v2.core.intelligence.v3.single_loop_managed_delivery import managed_record_matches
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimSuperseded
from alphonse.agent_v2.evaluation.synthetic_delivery_settlement import settle_fixture_delivery
from alphonse.agent_v2.evaluation.synthetic_staged_dispatch import dispatch_fixture_staged


def seed_authority(f):
    path, _, _, _, _, _, _ = f
    with sqlite3.connect(path) as conn:
        conn.execute('''CREATE TABLE fixture_delivery_grants(actor TEXT,recipient TEXT,integration TEXT,
                     provider TEXT,channel TEXT,grant_id TEXT,revision INTEGER,connection_revision INTEGER,active INTEGER)''')
        conn.execute('''CREATE TABLE fixture_delivery_effects(task_id TEXT,actor TEXT,action_id TEXT,
                     integration TEXT,provider TEXT,channel TEXT,recipient TEXT,message_digest TEXT,
                     idempotency_key TEXT,grant_id TEXT,grant_revision INTEGER,connection_revision INTEGER,status TEXT)''')
        conn.execute('INSERT INTO fixture_delivery_grants VALUES (?,?,?,?,?,?,?,?,?)',
                     ('alice', 'alice', 'desktop', 'desktop', 'alice', 'grant', 1, 1, 1))


def dispatch(f, staged_id, claim=None, clock=None):
    _, _, questions, outbox, original_claim, version, now = f
    return dispatch_fixture_staged(outbox, ExperimentalCheckpointStore(questions), claim or original_claim,
        actor='alice', task_id='task', expected_version=version, staged_id=staged_id, clock=clock or (lambda: now))


def settle(f, claim=None, clock=None):
    _, _, questions, _, original_claim, version, now = f
    return settle_fixture_delivery(ExperimentalCheckpointStore(questions), claim or original_claim,
        actor='alice', task_id='task', expected_version=version, clock=clock or (lambda: now))


def add_managed_preflight(f):
    # Even a structurally valid record cannot turn a synthetic fixture grant
    # into current managed authority or a real provider receipt.
    with sqlite3.connect(f[0]) as conn:
        task = json.loads(conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0])
        action = task['metadata']['single_loop_delivery_action']
        action['authority_mode'] = 'managed'
        binding = DeliveryBinding(**{key: action[key] for key in DeliveryBinding.__dataclass_fields__})
        task['metadata']['single_loop_managed_preflight'] = {
            'schema_version': 1, 'actor': 'alice', 'task_id': 'task', 'action_id': binding.action_id,
            'skill_id': 'skill:fixture', 'role': 'messages', 'binding_id': 'binding', 'binding_revision': 1,
            'required_scopes': ['scope.send'], 'required_executables': [], 'operation': 'send',
            'resources': [binding.channel_target], 'project_id': None, 'recipient_user_ids': ['alice'],
            'connection_id': binding.integration_id, 'provider_key': binding.provider_key,
            'channel_target': binding.channel_target, 'message_digest': binding.message_digest,
            'idempotency_key': binding.idempotency_key, 'grant_id': binding.grant_id,
            'grant_revision': binding.grant_revision, 'connection_revision': binding.connection_revision,
        }
        assert managed_record_matches(task, task_id='task', actor='alice', binding=binding)
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(task),))


def test_fixture_grant_cannot_dispatch_managed_action(setup):
    f = setup
    seed_authority(f)
    identity = stage(f)
    add_managed_preflight(f)
    with pytest.raises(ClaimFenceUnavailable, match='managed_unsupported'):
        dispatch(f, identity)
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 0
    assert ExperimentalCheckpointStore(f[2]).load('task', 'alice', expected_version=f[5]).state.inflight == 'action'


def test_fixture_effect_cannot_settle_managed_action(setup):
    f = setup
    seed_authority(f)
    identity = stage(f)
    dispatch(f, identity)
    add_managed_preflight(f)
    with pytest.raises(ClaimFenceUnavailable, match='managed_unsupported'):
        settle(f)
    assert ExperimentalCheckpointStore(f[2]).load('task', 'alice', expected_version=f[5]).state.inflight == 'action'
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 1


def test_stripped_managed_preflight_cannot_use_fixture_dispatch_or_settlement(setup):
    f = setup
    seed_authority(f)
    identity = stage(f)
    dispatch(f, identity)
    # Simulate a managed action whose separate preflight was lost. The action's
    # mode must prevent fixture evidence from being treated as its authority.
    with sqlite3.connect(f[0]) as conn:
        task = json.loads(conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0])
        task['metadata']['single_loop_delivery_action']['authority_mode'] = 'managed'
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(task),))
    with pytest.raises(ClaimFenceUnavailable):
        dispatch(f, identity)
    with pytest.raises(ClaimFenceUnavailable):
        settle(f)
    assert ExperimentalCheckpointStore(f[2]).load('task', 'alice', expected_version=f[5]).state.inflight == 'action'


def test_staged_fixture_dispatch_and_authorized_settlement_survive_restart(setup):
    f = setup
    seed_authority(f)
    identity = stage(f)
    dispatch(f, identity)
    # A lost dispatch response must not trigger a second fixture effect.
    with pytest.raises(ClaimFenceUnavailable):
        dispatch(f, identity)
    # Settlement uses a fresh checkpoint store, simulating a restarted worker;
    # it must independently read the already persisted fixture effect.
    settled = settle(f)
    restored = ExperimentalCheckpointStore(f[2]).load('task', 'alice', expected_version=settled.version)
    assert not restored.state.inflight
    assert restored.budget.to_dict()['reservations']['action']['outcome'] == 'success'
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 1
        assert conn.execute('SELECT status FROM v2_outbox WHERE outbox_message_id=?', (identity,)).fetchone()[0] == 'staged'


def test_restaging_after_fixture_effect_does_not_prove_non_dispatch(setup):
    f = setup
    seed_authority(f)
    identity = stage(f)
    dispatch(f, identity)
    # The inert row is unchanged by the fixture provider. Returning its ID
    # establishes matching staging only; it must not authorize another send.
    assert stage(f) == identity
    with pytest.raises(ClaimFenceUnavailable, match='already_observed'):
        dispatch(f, identity)
    assert ExperimentalCheckpointStore(f[2]).load('task', 'alice', expected_version=f[5]).state.inflight == 'action'
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 1


def test_wrong_staged_identity_never_dispatches(setup):
    f = setup
    seed_authority(f)
    stage(f)
    with pytest.raises(ClaimFenceUnavailable):
        dispatch(f, 'not-the-staged-message')
    assert ExperimentalCheckpointStore(f[2]).load('task', 'alice', expected_version=f[5]).state.inflight == 'action'
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 0


def test_revoked_before_dispatch_produces_no_effect_or_settlement(setup):
    f = setup
    seed_authority(f)
    identity = stage(f)
    with sqlite3.connect(f[0]) as conn:
        conn.execute('UPDATE fixture_delivery_grants SET active=0')
    with pytest.raises(ClaimFenceUnavailable):
        dispatch(f, identity)
    with pytest.raises(ClaimFenceUnavailable):
        settle(f)
    assert ExperimentalCheckpointStore(f[2]).load('task', 'alice', expected_version=f[5]).state.inflight == 'action'
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 0


def test_revocation_after_effect_preserves_uncertain_marker(setup):
    f = setup
    seed_authority(f)
    identity = stage(f)
    dispatch(f, identity)
    with sqlite3.connect(f[0]) as conn:
        conn.execute('UPDATE fixture_delivery_grants SET active=0')
    with pytest.raises(ClaimFenceUnavailable):
        settle(f)
    assert ExperimentalCheckpointStore(f[2]).load('task', 'alice', expected_version=f[5]).state.inflight == 'action'
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 1


def test_reclaimed_claim_cannot_dispatch_or_settle_old_worker(setup):
    f = setup
    seed_authority(f)
    identity = stage(f)
    assert f[1].reclaim_expired(now=f[6] + timedelta(days=1)) == 0
    simulate_old_writer_requeue(f[1], 'message')
    assert f[1].claim_next().claim_receipt.claim_generation != f[4].claim_generation
    with pytest.raises(ClaimSuperseded):
        dispatch(f, identity)
    with pytest.raises(ClaimSuperseded):
        settle(f)
    assert ExperimentalCheckpointStore(f[2]).load('task', 'alice', expected_version=f[5]).state.inflight == 'action'
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 0


def test_reclaim_after_effect_blocks_stale_settlement_without_replay(setup):
    f = setup
    seed_authority(f)
    identity = stage(f)
    dispatch(f, identity)
    assert f[1].reclaim_expired(now=f[6] + timedelta(days=1)) == 0
    simulate_old_writer_requeue(f[1], 'message')
    new_claim = f[1].claim_next().claim_receipt
    assert new_claim.claim_generation != f[4].claim_generation
    with pytest.raises(ClaimSuperseded):
        settle(f)
    with pytest.raises(ClaimFenceUnavailable):
        dispatch(f, identity, claim=new_claim)
    assert ExperimentalCheckpointStore(f[2]).load('task', 'alice', expected_version=f[5]).state.inflight == 'action'
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 1
