"""Synthetic decisions and disposable scoped acquisition; no live Jev/providers."""
from copy import deepcopy
import json

import pytest

from test_source_acquisition import acquisition, record
from test_project_check_scheduling import scheduled
from test_project_checks import checks
from test_account_grants import f
from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.integrations.acquisition import SourcePage
from alphonse.agent_v2.project_triage import decide, evaluate, VERSION, CONTRACT_DIGEST
from alphonse.agent_v2.services.project_triage import ProjectTriage
from alphonse.agent_v2.system_one import JevCriterionDecisionProvider, SystemOneSettings


def answers(relevance=.95, commitment=.95, change='unknown', urgency=0):
    return dict(relevance=dict(type='noul', noul=relevance), commitment=dict(type='noul', noul=commitment),
        change=dict(type='choice', choice=change, probabilities={k: float(k == change) for k in ('changed', 'unchanged', 'unknown')}),
        urgency=dict(type='score', score=urgency, confidence=.95,
                     probabilities={str(k): float(k == urgency) for k in range(3)}))


# Held-out response vectors: boundary confidence, correction, old information,
# explicit obligation, missing comparison and mixed relevance. No keyword parser
# pretending to validate a live semantic model.
@pytest.mark.parametrize('vector,expected', [
    (answers(), 'interpret'), (answers(.05, .05, 'unchanged'), 'irrelevant'),
    (answers(.95, .05, 'unchanged'), 'informational'),
    (answers(.95, .05, 'unknown'), 'pending'), (answers(.5), 'pending'),
    (answers(.95, .05, 'changed'), 'interpret'), (answers(.95, .05, 'unchanged', 2), 'interpret'),
    (answers(.79, .95), 'pending'), (answers(.8, .8), 'interpret'),
    (answers(.21, .05), 'pending'), (answers(.2, .05), 'irrelevant'),
])
def test_held_out_routing_vectors(vector, expected):
    assert decide(vector) == expected


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, 1.1, True, '0.9', None])
def test_invalid_probabilities(value):
    value_answers = answers()
    value_answers['relevance']['noul'] = value
    with pytest.raises(ValueError): decide(value_answers)


@pytest.mark.parametrize('mutation', ['missing', 'negative', 'nan', 'inconsistent', 'choice', 'confidence'])
def test_invalid_score_or_distribution(mutation):
    a = answers()
    if mutation == 'missing': a['urgency']['probabilities'].pop('1')
    if mutation == 'negative': a['urgency']['probabilities']['0'] = -1
    if mutation == 'nan': a['urgency']['score'] = float('nan')
    if mutation == 'inconsistent': a['urgency']['score'] = 2
    if mutation == 'choice': a['change']['choice'] = 'send_email'
    if mutation == 'confidence': a['urgency']['confidence'] = True
    with pytest.raises((ValueError, KeyError)): decide(a)


class Client:
    def __init__(self, vector=None): self.vector = vector or answers(); self.calls = []
    def evaluate(self, **kw):
        self.calls.append(kw)
        return dict(answers={key: deepcopy(self.vector[key.split('_', 1)[1]]) for key in kw['questions']},
                    usage={'total_tokens': 42, 'cost_usd': .0001, 'private': 'secret'})


def provider(client):
    p = JevCriterionDecisionProvider(SystemOneSettings(enabled=True, api_key='synthetic'))
    def transport(url, key, payload, timeout):
        return client.evaluate(state=payload['state'], questions=payload['questions'])
    p.client.transport = transport
    return p


def test_versioned_parallel_contract_and_content_free_telemetry():
    client = Client()
    results, telemetry = evaluate(client, goal={'query': 'school'}, candidates=[{'content': 'PRIVATE'}] * 2)
    assert results == ['interpret'] * 2
    assert len(client.calls[0]['questions']) == 8
    assert telemetry['version'] == VERSION and telemetry['contract_digest'] == CONTRACT_DIGEST
    assert telemetry['usage'] == {'total_tokens': 42, 'cost_usd': .0001}
    assert 'PRIVATE' not in json.dumps(telemetry) and 'secret' not in json.dumps(telemetry)


def setup(f, client=None):
    f.client = client or Client()
    f.acquisition.triage = ProjectTriage(f.acquisition, lambda: provider(f.client))
    f.adapter.pages = [SourcePage((record(f),), 'history', True)]
    assert f.process()['state'] == 'pending'
    return f


def test_exact_scoped_receipt_and_processing_checkpoint(acquisition):
    f = setup(acquisition)
    assert f.process()['state'] == 'pending'
    saved = f.acquisition.evidence(AccountSession('child'), f.run_id)[0]
    receipt = f.acquisition.triage.receipts(AccountSession('child'), f.run_id)[0]
    assert receipt['evidence_id'] == saved['evidence_id']
    assert receipt['content_digest'] == saved['content_digest']
    assert receipt['source_version'] == saved['version']
    assert receipt['status'] == 'interpret'
    assert f.acquisition.processing(AccountSession('child'), f.run_id)['classification'] == [saved['evidence_id']]
    assert f.process()['outcome'] == 'complete'
    assert len(f.client.calls) == 1 and len(f.adapter.calls) == 1
    with pytest.raises(PermissionError): f.acquisition.triage.receipts(AccountSession('parent'), f.run_id)


@pytest.mark.parametrize('kind', ['ambiguous', 'outage', 'disabled', 'saturation', 'oversize', 'invalid'])
def test_uncertainty_retained_without_llm_fallback(acquisition, kind):
    f = setup(acquisition, Client(answers(.5) if kind == 'ambiguous' else answers()))
    if kind == 'outage':
        def fail(**kw): raise RuntimeError('private provider error')
        f.client.evaluate = fail
    if kind == 'disabled': f.acquisition.triage.provider = lambda: None
    if kind == 'saturation':
        with f.schedules._connect() as conn:
            raw = json.loads(conn.execute('SELECT receipt FROM v2_project_check_runs WHERE check_run_id=?', (f.run_id,)).fetchone()[0])
            raw['model_calls'] = 2
            conn.execute('UPDATE v2_project_check_runs SET receipt=? WHERE check_run_id=?', (json.dumps(raw), f.run_id))
    if kind == 'oversize':
        with f.schedules._connect() as conn:
            raw = json.loads(conn.execute('SELECT definition FROM v2_source_evidence').fetchone()[0])
            raw['content']['parts'][0]['text'] = 'x' * 5000
            conn.execute('UPDATE v2_source_evidence SET definition=?', (json.dumps(raw),))
    if kind == 'invalid': f.client.vector['urgency']['score'] = float('nan')
    assert f.process()['state'] == 'pending'
    assert f.process()['outcome'] == 'partial'
    receipt = f.acquisition.triage.receipts(AccountSession('child'), f.run_id)[0]
    assert receipt['status'] == 'pending'
    assert f.acquisition.processing(AccountSession('child'), f.run_id)['classification'] == []
    assert 'private provider error' not in json.dumps(receipt)
    assert len(f.adapter.calls) == 1


def test_revoke_during_triage_cannot_commit(acquisition):
    f = setup(acquisition)
    original = f.client.evaluate
    def revoke(**kw):
        f.service.revoke_grant(AccountSession('parent'), grant_id='school-grant', expected_revision=1)
        return original(**kw)
    f.client.evaluate = revoke
    assert f.process()['outcome'] == 'blocked'
    with f.schedules._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM v2_project_triage').fetchone()[0] == 0


def test_managed_capability_exposure_and_forged_execution(f):
    manifest = f.bindings.account_capabilities(AccountSession('child'), skill_id='skill:fixture')
    assert manifest and manifest[0]['operation'] == 'read'
    assert 'secret' not in json.dumps(manifest) and 'token' not in json.dumps(manifest)
    assert not f.bindings.account_capabilities(AccountSession('other'), skill_id='skill:fixture')
    request = dict(skill_id='skill:fixture', name='mail', operation='send', resources=['folder-a'], recipient_user_ids=[])
    called = []
    with pytest.raises(PermissionError):
        f.bindings.use_account(AccountSession('child'), request=request, operation=lambda *args: called.append(args))
    assert not called
    f.service.revoke_grant(AccountSession('parent'), grant_id='grant-one', expected_revision=1)
    assert not f.bindings.account_capabilities(AccountSession('child'), skill_id='skill:fixture')


def test_batch_saturation_is_bounded_and_never_discards_candidates(acquisition):
    from dataclasses import replace
    f = acquisition
    f.client = Client()
    f.acquisition.triage = ProjectTriage(f.acquisition, lambda: provider(f.client))
    f.adapter.pages = [SourcePage(tuple(replace(record(f), source_id=f'message-{i}') for i in range(33)), 'history', True)]
    assert f.process()['state'] == 'pending'
    for _ in range(3): assert f.process()['state'] == 'pending'
    assert f.process()['outcome'] == 'partial'
    receipts = f.acquisition.triage.receipts(AccountSession('child'), f.run_id)
    assert len(receipts) == 33
    assert sum(r['status'] == 'pending' for r in receipts) == 1
    assert len(f.client.calls) == 2
    assert all(len(c['questions']) == 64 for c in f.client.calls)
    assert f.dispatcher.get(f.run_id)['model_calls'] == 2


def test_confident_irrelevance_stays_quiet_preserving_acquisition_counts(acquisition):
    f = setup(acquisition, Client(answers(.05, .05, 'unchanged')))
    f.process()
    final = f.process()
    assert final['outcome'] == 'complete' and final['items'] == 1
    assert final['coverage']['triage_no_match']
    with f.events._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM v2_event_history').fetchone()[0] == 0


def test_cancel_between_acquisition_and_triage_prevents_model_work(acquisition):
    f = setup(acquisition)
    f.dispatcher.cancel(f.run_id, actor_user_id='child', now=f.now)
    assert f.process() is None
    assert not f.client.calls
    with pytest.raises(PermissionError):
        f.acquisition.triage.receipts(AccountSession('child'), f.run_id)


def test_state_budget_fails_atomically_before_model_call():
    client = Client()
    with pytest.raises(ValueError, match='content_budget'):
        evaluate(client, goal={'text': 'x' * 65536}, candidates=[{}])
    with pytest.raises(ValueError, match='batch_budget'):
        evaluate(client, goal={}, candidates=[{}] * 17)
    assert not client.calls


def test_crash_after_receipt_recovers_checkpoint_without_inference(acquisition, monkeypatch):
    from datetime import timedelta
    f = setup(acquisition)
    checkpoint = f.acquisition.checkpoint_processing
    calls = []
    def crash(*args, **kw):
        calls.append(kw)
        if len(calls) == 2:
            raise SystemExit('after receipt before classification checkpoint')
        return checkpoint(*args, **kw)
    monkeypatch.setattr(f.acquisition, 'checkpoint_processing', crash)
    with pytest.raises(SystemExit): f.process()
    f.now += timedelta(seconds=31)
    monkeypatch.setattr(f.acquisition, 'checkpoint_processing', checkpoint)
    assert f.process()['outcome'] == 'complete'
    assert len(f.client.calls) == 1
    assert len(f.acquisition.processing(AccountSession('child'), f.run_id)['classification']) == 1


def test_activated_skill_context_exposes_only_current_managed_choices(f):
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3.context import selected_skill_guidance
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    task = TaskState(task_id='triage-test', user='child')
    task.metadata['v3_skill_selection'] = {'selected_skill_ids': ['skill:fixture']}
    context = CoreLoopContext(messages=InMemoryMessageQueue(), skill_store=f.skills, skill_bindings=f.bindings)
    text, _ = selected_skill_guidance(task, context)
    assert 'Managed account capabilities' in text and '"operation": "read"' in text
    assert '"credentials"' not in text
    f.service.revoke_grant(AccountSession('parent'), grant_id='grant-one', expected_revision=1)
    text, _ = selected_skill_guidance(task, context)
    assert 'Managed account capabilities' not in text
