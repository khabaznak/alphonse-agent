import concurrent.futures

import pytest

from alphonse.agent_v2.coding_jobs import CodingJobStore


def test_reserve_replays_only_the_same_request_and_is_actor_scoped(tmp_path):
    store = CodingJobStore(tmp_path / 'jobs.sqlite')
    request = dict(actor='owner', task_id='task', operation_id='operation', request_hash='a' * 64)
    first = store.reserve(**request)
    assert store.reserve(**request) == first
    assert store.get(job_id=first['job_id'], actor='other', task_id='task') is None
    with pytest.raises(ValueError, match='operation_conflict'):
        store.reserve(**{**request, 'request_hash': 'b' * 64})
    assert store.reserve(**{**request, 'actor': 'other'})['job_id'] != first['job_id']


def test_only_one_concurrent_worker_can_claim_a_job(tmp_path):
    store = CodingJobStore(tmp_path / 'jobs.sqlite')
    job = store.reserve(actor='owner', task_id='task', operation_id='operation', request_hash='a' * 64)
    def claim():
        try:
            return store.transition(job_id=job['job_id'], actor='owner', task_id='task', expected='queued', state='running')['state']
        except ValueError:
            return 'conflict'
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: claim(), range(2))) == ['conflict', 'running']
    interrupted = store.transition(job_id=job['job_id'], actor='owner', task_id='task', expected='running', state='interrupted')
    assert interrupted['state'] == 'interrupted'
    with pytest.raises(ValueError, match='transition_invalid'):
        store.transition(job_id=job['job_id'], actor='owner', task_id='task', expected='interrupted', state='running')


def test_result_codes_do_not_store_provider_output(tmp_path):
    store = CodingJobStore(tmp_path / 'jobs.sqlite')
    job = store.reserve(actor='owner', task_id='task', operation_id='op', request_hash='a' * 64)
    store.transition(job_id=job['job_id'], actor='owner', task_id='task', expected='queued', state='running')
    with pytest.raises(ValueError, match='result_code_invalid'):
        store.transition(job_id=job['job_id'], actor='owner', task_id='task', expected='running', state='failed', result_code='sensitive output')
    done = store.transition(job_id=job['job_id'], actor='owner', task_id='task', expected='running', state='failed', result_code='provider_failed')
    assert done['result_code'] == 'provider_failed'
