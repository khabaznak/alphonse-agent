from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from hashlib import sha256
from threading import Event

from alphonse.agent_v2.core.intelligence.simple_loop.owner_disposable_handoff import (
    DisposableHandoffBinding,
    DisposableHandoffReceipt,
    DisposableHandoffService,
)


PROMPT = 'Keep the workshop note private. Ask Codex to review the fixture repository.'
BINDING = DisposableHandoffBinding('owner', 'fixture-repo', 'revision-1', 'review', sha256(PROMPT.encode()).hexdigest())
SCOPE = {'actor': 'owner', 'parent_id': 'parent', 'target': 'fixture-repo', 'worker': 'codex', 'operation': 'review',
         'owner_prompt': PROMPT}


def make_service(path, *, submit=lambda *_: None, reconcile=lambda *_: None, verify=lambda *_: False,
                 bindings=(BINDING,)):
    return DisposableHandoffService(path, bindings=bindings, submit=submit, reconcile=reconcile,
                                    verify_result=verify)


def dispatch(service, **overrides):
    return service.dispatch(**(SCOPE | {'claim_active': lambda: True} | overrides))


def test_fixture_authorization_and_worker_projection(tmp_path):
    submitted = []

    def submit(request, claim_active):
        assert claim_active()
        submitted.append(asdict(request))
        return DisposableHandoffReceipt(request, 'accepted')

    service = make_service(tmp_path / 'handoff.sqlite', submit=submit)
    assert service.targets_for(actor='owner', owner_prompt=PROMPT, claim_active=lambda: True) == (('fixture-repo', 'review'),)
    for overrides in ({'actor': 'other'}, {'worker': 'other'}, {'target': 'other'}, {'operation': 'propose_patch'},
                      {'owner_prompt': 'Could you review it?'}, {'claim_active': lambda: False}):
        refused = dispatch(service, **overrides)
        assert refused['status'] == 'blocked' and refused['operation_id'] is None
        assert refused['target'] is None
    assert submitted == []

    accepted = dispatch(service)
    assert accepted['status'] == 'accepted' and accepted['effect_status'] == 'not_requested'
    assert submitted == [{'operation_id': accepted['operation_id'], 'worker': 'codex', 'target': 'fixture-repo',
                          'revision': 'revision-1', 'operation': 'review'}]
    assert service.valid_observation(actor='owner', parent_id='parent', observation=accepted)
    assert not service.valid_observation(actor='other', parent_id='parent', observation=accepted)
    assert not service.valid_observation(actor='owner', parent_id='different', observation=accepted)
    assert not service.valid_observation(actor='owner', parent_id='parent',
                                         observation=accepted | {'effect_status': 'applied'})


def test_reconcile_after_uncertain_dispatch_and_pin_revision(tmp_path):
    path = tmp_path / 'handoff.sqlite'
    submitted, reconciled = [], []

    def submit(request, _):
        submitted.append(request)
        raise TimeoutError('receipt lost')

    def reconcile(request, _):
        reconciled.append(request)
        return DisposableHandoffReceipt(request, 'completed')

    unknown = dispatch(make_service(path, submit=submit))
    assert unknown['status'] == 'unknown'

    changed = make_service(path, submit=submit, reconcile=reconcile, bindings=(replace(BINDING, revision='revision-2'),))
    assert dispatch(changed)['result_code'] == 'scope_conflict'
    assert not changed.valid_observation(actor='owner', parent_id='parent', observation=unknown)

    restarted = make_service(path, submit=submit, reconcile=reconcile,
                             verify=lambda req, _, active: active() and req in submitted)
    completed = dispatch(restarted)
    assert completed['status'] == 'completed' and completed['result_code'] == 'result_verified'
    assert completed['operation_id'] == unknown['operation_id']
    assert completed['effect_status'] == 'not_requested'
    assert restarted.valid_observation(actor='owner', parent_id='parent', observation=completed)
    assert dispatch(restarted) == completed
    assert submitted == reconciled and len(submitted) == 1


def test_forged_receipt_and_unverified_completion_cannot_advance(tmp_path):
    path = tmp_path / 'handoff.sqlite'
    service = make_service(path, submit=lambda req, _: DisposableHandoffReceipt(replace(req, revision='wrong'), 'accepted'),
                           reconcile=lambda req, _: DisposableHandoffReceipt(req, 'completed'))
    result = dispatch(service)
    assert result['status'] == 'unknown'
    assert dispatch(service) == result
    false_completion = result | {'status': 'completed', 'result_code': 'result_verified'}
    assert not service.valid_observation(actor='owner', parent_id='parent', observation=false_completion)


def test_claim_loss_withholds_worker_receipt_until_reconciliation(tmp_path):
    claim = True
    submitted = []

    def submit(request, claim_active):
        nonlocal claim
        assert claim_active()
        submitted.append(request)
        claim = False
        return DisposableHandoffReceipt(request, 'accepted')

    service = make_service(tmp_path / 'handoff.sqlite', submit=submit,
                           reconcile=lambda req, active: DisposableHandoffReceipt(req, 'accepted') if active() else None)
    uncertain = dispatch(service, claim_active=lambda: claim)
    assert uncertain['status'] == 'unknown'
    assert dispatch(service, claim_active=lambda: claim)['result_code'] == 'claim_unavailable'
    recovered = dispatch(service)
    assert recovered['operation_id'] == uncertain['operation_id']
    assert recovered['status'] == 'accepted' and len(submitted) == 1


def test_concurrent_lookup_never_submits_operation_twice(tmp_path):
    submitted = []
    started, release = Event(), Event()

    def submit(request, _):
        submitted.append(request)
        started.set()
        if release.wait(5):
            return DisposableHandoffReceipt(request, 'accepted')
        return None

    service = make_service(tmp_path / 'handoff.sqlite', submit=submit)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_dispatch = pool.submit(dispatch, service)
        try:
            assert started.wait(5)
            second = pool.submit(dispatch, service).result(timeout=5)
            assert second['status'] == 'unknown'
        finally:
            release.set()
        first = first_dispatch.result(timeout=5)
    assert first['status'] == 'accepted' and first['operation_id'] == second['operation_id']
    assert len(submitted) == 1
