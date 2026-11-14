"""Synthetic candidate model step with an explicitly supplied queue claim."""
from dataclasses import asdict, replace
from datetime import timedelta

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.v3.single_loop_atomic_step import run_atomic_step
from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateHost
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from test_experimental_claim_checkpoint import setup
from unsafe_reclaim_fixture import simulate_old_writer_requeue


@pytest.fixture
def host(setup):
    _, _, questions, _, _, _, now = setup
    task = questions.load_task_checkpoint('task')
    calls = []
    def choose(prompt, cap):
        calls.append((prompt, cap))
        return {'kind': 'tool', 'tool_id': 'synthetic', 'arguments': {},
                'usage': {'input_tokens': 10, 'output_tokens': 5}}
    def forbidden(*args):
        pytest.fail('tool, evidence, or publication was unexpectedly invoked')
    result = CandidateHost(task=task, context=CoreLoopContext(messages=InMemoryMessageQueue()), budgets=None,
        checkpoint=forbidden, choose=choose, authorize_source=lambda actor, ref: actor == 'alice' and ref == 'goal',
        load_source=lambda actor, ref: 'Synthetic request', authorize_read=forbidden,
        record_evidence=forbidden, verify_response=forbidden, now_ms=lambda: int(now.timestamp() * 1000),
        input_reservation=400, output_reservation=50)
    return result, calls


def execute(f, host, **kwargs):
    _, _, _, store, claim, version, now = f
    return run_atomic_step(store.load('task', 'alice', expected_version=version).state, host, store,
                           expected_version=version, claim=kwargs.pop('claim', claim),
                           claim_clock=kwargs.pop('claim_clock', lambda: now), **kwargs)


def test_live_claim_fences_reservation_and_model_settlement(setup, host):
    _, _, questions, store, claim, version, _ = setup
    result = execute(setup, host[0])
    assert result.reason == 'tool_dispatch_and_evidence_not_integrated'
    assert result.checkpoint_version == version + 2 and len(host[1]) == 1
    checkpoint = ExperimentalCheckpointStore(questions).load('task', 'alice', expected_version=result.checkpoint_version)
    assert not checkpoint.state.inflight and not checkpoint.budget.usage()['in_flight']
    assert claim.claim_generation not in str(questions.load_task_checkpoint('task').metadata)
    assert claim.claim_generation not in str(asdict(result))


def test_claim_loss_during_model_retains_marker_and_prevents_settlement(setup, host):
    queue, _, questions, store, old, version, _ = setup
    def choose(*args):
        simulate_old_writer_requeue(queue, 'message')
        assert queue.claim_next().claim_receipt.claim_generation != old.claim_generation
        return {'kind': 'answer', 'text': 'synthetic', 'usage': {'input_tokens': 10, 'output_tokens': 5}}
    result = execute(setup, replace(host[0], choose=choose))
    assert result.reason == 'model_settlement_unavailable'
    checkpoint = store.load('task', 'alice', expected_version=version + 1)
    assert checkpoint.state.inflight.startswith('model-')
    assert old.claim_generation not in str(questions.load_task_checkpoint('task').metadata)


def test_reclaimed_claim_never_reads_sources_or_calls_model(setup, host):
    queue, _, _, store, old, version, _ = setup
    simulate_old_writer_requeue(queue, 'message')
    assert queue.claim_next().claim_receipt.claim_generation != old.claim_generation
    reads = []
    source_host = replace(host[0],
        authorize_source=lambda actor, ref: reads.append(('authorize', actor, ref)) or True,
        load_source=lambda actor, ref: reads.append(('load', actor, ref)) or 'synthetic')

    result = execute(setup, source_host)

    assert result.reason == 'checkpoint_projection_or_budget_unavailable'
    assert reads == [] and host[1] == []
    assert not store.load('task', 'alice', expected_version=version).state.inflight


def test_expired_claim_never_reads_sources_or_calls_model(setup, host):
    _, _, _, store, _, version, now = setup
    reads = []
    source_host = replace(host[0],
        authorize_source=lambda actor, ref: reads.append(('authorize', actor, ref)) or True,
        load_source=lambda actor, ref: reads.append(('load', actor, ref)) or 'synthetic')

    result = execute(setup, source_host, claim_clock=lambda: now + timedelta(days=1))

    assert result.reason == 'checkpoint_projection_or_budget_unavailable'
    assert reads == [] and host[1] == []
    assert not store.load('task', 'alice', expected_version=version).state.inflight


def test_crash_or_cancel_keeps_claim_bound_reservation(setup, host):
    _, _, questions, store, _, version, _ = setup
    def crash(*args):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        execute(setup, replace(host[0], choose=crash))
    pending = store.load('task', 'alice', expected_version=version + 1)
    assert pending.state.inflight.startswith('model-')
    assert run_atomic_step(pending.state, host[0], store, expected_version=pending.version,
                           claim=setup[4], claim_clock=lambda: setup[6]).reason == 'inflight_requires_reconciliation'
    assert not host[1]
    assert questions.mark_task_checkpoint_terminal('task', status='cancelled')
    assert store.inspect_cancelled('task', 'alice', expected_version=pending.version)['inflight'] == pending.state.inflight


def test_expired_claim_before_dispatch_blocks_without_model_call(setup, host):
    _, _, _, store, _, version, now = setup
    # Source preflight checks twice and reservation checks three times; expiry
    # is observed by the next pre-dispatch transaction.
    samples = iter([now] * 5 + [now + timedelta(days=1)])
    result = execute(setup, host[0], claim_clock=lambda: next(samples))
    assert result.reason == 'checkpoint_changed_before_model'
    assert not host[1]
    assert store.load('task', 'alice', expected_version=version + 1).state.inflight.startswith('model-')


def test_lease_expires_after_fenced_precheck_before_model_call(setup, host):
    _, _, _, store, _, version, now = setup
    # The final clock check occurs after the fenced transaction and host checks;
    # it must still reject expiry before the chooser is called.
    samples = iter([now] * 7 + [now + timedelta(seconds=61)])
    result = execute(setup, host[0], claim_clock=lambda: next(samples))
    assert result.reason == 'deadline_or_clock_invalid_before_model'
    assert not host[1]
    assert store.load('task', 'alice', expected_version=version + 1).state.inflight.startswith('model-')


def test_incomplete_claim_configuration_never_uses_unfenced_fallback(setup, host):
    _, _, _, store, claim, version, _ = setup
    state = store.load('task', 'alice', expected_version=version).state
    result = run_atomic_step(state, host[0], store, expected_version=version, claim=claim)
    assert result.reason == 'claim_configuration_invalid'
    assert not host[1] and not store.load('task', 'alice', expected_version=version).state.inflight


def test_tampered_boolean_claim_schema_never_reaches_model(setup, host):
    _, _, questions, store, _, version, now = setup
    def tamper():
        import json
        with questions._connect() as conn:
            row = conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()
            raw = json.loads(row[0])
            raw['metadata']['single_loop_model_claim']['schema_version'] = True
            conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(raw),))
        return int(now.timestamp() * 1000)
    result = execute(setup, replace(host[0], now_ms=tamper))
    assert result.reason == 'checkpoint_changed_before_model'
    assert not host[1]
    assert store.load('task', 'alice', expected_version=version + 1).state.inflight.startswith('model-')


def test_actor_deactivation_before_dispatch_never_reaches_model(setup, host):
    _, users, _, store, _, version, now = setup
    def deactivate():
        users.update_user('alice', is_active=False)
        return int(now.timestamp() * 1000)
    result = execute(setup, replace(host[0], now_ms=deactivate))
    assert result.reason == 'checkpoint_changed_before_model'
    assert not host[1]
    assert store.load('task', 'alice', expected_version=version + 1).state.inflight.startswith('model-')


def test_cancellation_during_model_keeps_claim_bound_reservation(setup, host):
    _, _, questions, store, _, version, _ = setup
    def cancel(*args):
        assert questions.mark_task_checkpoint_terminal('task', status='cancelled')
        host[0].context.cancellation_checker = lambda: True
        return {'kind': 'answer', 'text': 'synthetic', 'usage': {'input_tokens': 10, 'output_tokens': 5}}
    result = execute(setup, replace(host[0], choose=cancel))
    assert (result.status, result.reason) == ('cancelled', 'cancelled_after_model')
    inspection = store.inspect_cancelled('task', 'alice', expected_version=version + 1)
    assert inspection['kind'] == 'model' and inspection['inflight'].startswith('model-')
