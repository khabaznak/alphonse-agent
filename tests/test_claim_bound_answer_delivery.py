"""Disposable model-to-delivery connection, not daemon or provider recovery."""
from dataclasses import replace
from datetime import timedelta
import hashlib
import sqlite3
import subprocess
import sys

import pytest

from alphonse.agent_v2.core.intelligence.v3.single_loop_atomic_step import run_atomic_step
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateHost
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import CheckpointConflict, ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.v3.single_loop_delivery_reservation import DeliveryBinding, reserve_delivery
from alphonse.agent_v2.core.io.experimental_staged_outbox import install_experimental_staging, stage_experimental_response
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.evaluation.synthetic_staged_dispatch import dispatch_fixture_staged
from test_claim_bound_atomic_step import host
from test_claim_bound_bootstrap import fixture as boot_fixture, bootstrap
from test_experimental_claim_checkpoint import setup
from test_synthetic_delivery_lifecycle import seed_authority
from unsafe_reclaim_fixture import simulate_old_writer_requeue
from alphonse.agent_v2.core.messages import InMemoryMessageQueue


def test_selected_answer_is_bound_through_fixture_effect_and_settlement(boot_fixture):
    queue, _, questions, store, claim, _, now = boot_fixture
    created = bootstrap(boot_fixture)
    task = questions.load_task_checkpoint('message')
    text = 'A synthetic answer.'
    def forbidden(*_):
        pytest.fail('unexpected fixture side effect')
    selected = CandidateHost(task=task, context=CoreLoopContext(messages=InMemoryMessageQueue()), budgets=None,
        checkpoint=forbidden,
        choose=lambda *_: {'kind': 'answer', 'text': text, 'usage': {'input_tokens': 10, 'output_tokens': 5}},
        authorize_source=lambda actor, ref: actor == 'alice' and ref == 'queued_goal',
        load_source=lambda actor, ref: task.goal, authorize_read=forbidden, record_evidence=forbidden,
        verify_response=lambda actor, answer, sources: actor == 'alice' and answer == text and not sources,
        now_ms=lambda: int(now.timestamp() * 1000), input_reservation=2000, output_reservation=50)
    result = run_atomic_step(created.state, selected, store, expected_version=created.version, claim=claim,
                             claim_clock=lambda: now, retain_fixture_answer=True)
    assert result.reason == 'fixture_answer_retained_not_delivered'
    # The output is durable, and the opaque raw acquisition token is not.
    raw = questions.load_task_checkpoint('message').metadata
    assert raw['fixture_selected_answer']['message'] == text
    assert claim.claim_generation not in str(raw)
    digest = hashlib.sha256(text.encode()).hexdigest()
    binding = DeliveryBinding('action', 'desktop', 'desktop', 'alice', 'alice', digest, 'key', 'grant', 1, 1)
    with pytest.raises(ValueError, match='message_mismatch'):
        reserve_delivery(store, claim, task_id='message', actor='alice', expected_version=result.checkpoint_version,
                         binding=binding, message='different', now_ms=int(now.timestamp() * 1000), clock=lambda: now)
    other = replace(binding, message_digest=hashlib.sha256(b'different').hexdigest())
    with pytest.raises(CheckpointConflict, match='selected_answer_mismatch'):
        reserve_delivery(store, claim, task_id='message', actor='alice', expected_version=result.checkpoint_version,
                         binding=other, message='different', now_ms=int(now.timestamp() * 1000), clock=lambda: now)
    reserved = reserve_delivery(store, claim, task_id='message', actor='alice', expected_version=result.checkpoint_version,
        binding=binding, message=text, now_ms=int(now.timestamp() * 1000), clock=lambda: now)
    outbox = SQLiteOutboundStore(questions.db_path)
    install_experimental_staging(outbox)
    f = (questions.db_path, queue, questions, outbox, claim, reserved.version, now)
    seed_authority(f)
    with sqlite3.connect(questions.db_path) as conn:
        conn.execute('CREATE TABLE fixture_recovery_grants(actor TEXT, task_id TEXT, active INTEGER)')
        conn.execute('INSERT INTO fixture_recovery_grants VALUES (?,?,?)', ('alice', 'message', 1))
    identity = stage_experimental_response(outbox, claim, task_id='message', actor='alice',
        expected_version=reserved.version, integration_id='desktop', provider_key='desktop', channel_target='alice',
        message=text, idempotency_key='key', clock=lambda: now, checkpoints=store)
    dispatch_fixture_staged(outbox, store, claim, actor='alice', task_id='message', expected_version=reserved.version,
                            staged_id=identity, clock=lambda: now)
    # A fresh process uses the durable expired queue row, without receiving the
    # original claim receipt. This is fixture reconciliation, not takeover.
    script = '''from datetime import datetime
import sys
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.evaluation.synthetic_delivery_settlement import recover_expired_fixture_effect
store = ExperimentalCheckpointStore(SQLiteQuestionStore(sys.argv[1]))
result = recover_expired_fixture_effect(store, actor='alice', task_id='message', expected_version=int(sys.argv[2]),
                                       clock=lambda: datetime.fromisoformat(sys.argv[3]))
print(result.version)
'''
    child = subprocess.run([sys.executable, '-c', script, questions.db_path, str(reserved.version),
                            (now + timedelta(days=1)).isoformat()], capture_output=True, text=True, check=True)
    assert child.stdout.strip() == str(reserved.version + 1) and child.stderr == ''
    settled = ExperimentalCheckpointStore(questions).load('message', 'alice', expected_version=reserved.version + 1)
    assert not settled.state.inflight
    with sqlite3.connect(questions.db_path) as conn:
        assert conn.execute('SELECT count(*) FROM experimental_staged_dispatch_attempts').fetchone()[0] == 1
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 1
    assert queue.status_for('message') == 'processing'  # no acknowledgement or replay


def test_unverified_answer_is_unaccepted_and_cannot_be_reserved(setup, host):
    _, _, questions, store, claim, version, now = setup
    selected = replace(host[0],
        choose=lambda *_: {'kind': 'answer', 'text': 'unsupported', 'usage': {'input_tokens': 10, 'output_tokens': 5}},
        verify_response=lambda *_: False)
    result = run_atomic_step(store.load('task', 'alice', expected_version=version).state, selected, store,
                             expected_version=version, claim=claim, claim_clock=lambda: now, retain_fixture_answer=True)
    assert result.reason == 'communication_and_completion_not_integrated'
    assert questions.load_task_checkpoint('task').metadata['fixture_selected_answer']['accepted'] is False
    binding = DeliveryBinding('action', 'desktop', 'desktop', 'alice', 'alice',
        hashlib.sha256(b'unsupported').hexdigest(), 'key', 'grant', 1, 1)
    with pytest.raises(CheckpointConflict, match='selected_answer_mismatch'):
        reserve_delivery(store, claim, task_id='task', actor='alice', expected_version=result.checkpoint_version,
                         binding=binding, message='unsupported', now_ms=int(now.timestamp() * 1000), clock=lambda: now)


def test_non_answer_fixture_decision_cannot_fall_back_to_generic_delivery(setup, host):
    _, _, questions, store, claim, version, now = setup
    result = run_atomic_step(store.load('task', 'alice', expected_version=version).state, host[0], store,
                             expected_version=version, claim=claim, claim_clock=lambda: now, retain_fixture_answer=True)
    assert result.reason == 'tool_dispatch_and_evidence_not_integrated'
    assert questions.load_task_checkpoint('task').metadata['fixture_selected_answer']['accepted'] is False
    binding = DeliveryBinding('action', 'desktop', 'desktop', 'alice', 'alice',
        hashlib.sha256(b'invented').hexdigest(), 'key', 'grant', 1, 1)
    with pytest.raises(CheckpointConflict, match='selected_answer_mismatch'):
        reserve_delivery(store, claim, task_id='task', actor='alice', expected_version=result.checkpoint_version,
                         binding=binding, message='invented', now_ms=int(now.timestamp() * 1000), clock=lambda: now)


def test_claim_loss_during_answer_verification_retains_model_marker(setup, host):
    queue, _, questions, store, claim, version, now = setup
    def verify(*_):
        simulate_old_writer_requeue(queue, 'message')
        queue.claim_next()
        return True
    selected = replace(host[0],
        choose=lambda *_: {'kind': 'answer', 'text': 'answer', 'usage': {'input_tokens': 10, 'output_tokens': 5}},
        verify_response=verify)
    result = run_atomic_step(store.load('task', 'alice', expected_version=version).state, selected, store,
                             expected_version=version, claim=claim, claim_clock=lambda: now, retain_fixture_answer=True)
    assert result.reason == 'model_settlement_unavailable'
    assert store.load('task', 'alice', expected_version=version + 1).state.inflight.startswith('model-')
    assert 'fixture_selected_answer' not in questions.load_task_checkpoint('task').metadata


def test_final_stale_evidence_cannot_be_delivered(setup, host, monkeypatch):
    from alphonse.agent_v2.core.intelligence.v3 import single_loop_atomic_step
    _, _, questions, store, claim, version, now = setup
    checks = iter((True, True, True, False))
    monkeypatch.setattr(single_loop_atomic_step, '_is_fresh', lambda *_: next(checks))
    text = 'answer'
    selected = replace(host[0],
        choose=lambda *_: {'kind': 'answer', 'text': text, 'usage': {'input_tokens': 10, 'output_tokens': 5}},
        verify_response=lambda *_: True)
    result = run_atomic_step(store.load('task', 'alice', expected_version=version).state, selected, store,
                             expected_version=version, claim=claim, claim_clock=lambda: now, retain_fixture_answer=True)
    assert result.reason == 'evidence_changed_or_revoked'
    assert questions.load_task_checkpoint('task').metadata['fixture_selected_answer']['accepted'] is False
    binding = DeliveryBinding('action', 'desktop', 'desktop', 'alice', 'alice',
        hashlib.sha256(text.encode()).hexdigest(), 'key', 'grant', 1, 1)
    with pytest.raises(CheckpointConflict, match='selected_answer_mismatch'):
        reserve_delivery(store, claim, task_id='task', actor='alice', expected_version=result.checkpoint_version,
                         binding=binding, message=text, now_ms=int(now.timestamp() * 1000), clock=lambda: now)
