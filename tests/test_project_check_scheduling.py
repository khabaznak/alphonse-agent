"""Synthetic project checks; no provider, token or model calls."""
from datetime import datetime, timedelta, timezone
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from test_account_grants import f
from test_project_checks import checks, active
from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore, compute_next_run_at
from alphonse.agent_v2.services.scheduled_worker import ScheduledTaskWorker
from alphonse.agent_v2.services.project_check_worker import (
    ProjectCheckDispatcher, ProjectCheckWorker, CheckStep, CheckContext, CheckStopped,
)


@pytest.fixture
def scheduled(checks):
    f = checks
    # A real-time Monday makes account expiry checks realistic even when run on
    # another weekday; only the deterministic occurrence clock is advanced.
    today = datetime.now(timezone.utc)
    monday = today.replace(hour=14, minute=0, second=0, microsecond=0)
    monday += timedelta(days=(0 - monday.weekday()) % 7)
    f.definition['duration'].update(starts_at=(today - timedelta(days=10)).isoformat(),
                                    ends_at=(today + timedelta(days=40)).isoformat())
    f.policy = active(f)
    f.now = monday
    f.schedules = ScheduledTaskStore(str(f.tmp / 'schedules.sqlite'))
    f.dispatcher = ProjectCheckDispatcher(f.schedules, f.checks)
    f.queue = InMemoryMessageQueue()
    f.scheduler = ScheduledTaskWorker(store=f.schedules, messages=f.queue, project_check_handler=f.dispatcher)
    f.dispatcher.sync(now=monday)
    f.task = f.schedules.list_tasks()[0]
    # Installation normally starts at authorization. Position at a known Monday
    # rather than depending on which day this fixture happens to run.
    f.task = replace(f.task, next_run_at=monday.isoformat())
    f.schedules._save_task(f.task)
    f.dispatch = lambda: f.scheduler.run_once(now=f.now)[0]
    f.worker = lambda adapter=None, **kw: ProjectCheckWorker(f.dispatcher, adapter=adapter, clock=lambda: f.now, **kw)
    return f


def one(ctx):
    return ctx.perform('page', lambda: CheckStep(acquisition_checkpoint={'page': 1}, complete=True))


def test_typed_dispatch_is_durable_idempotent_and_never_queues_llm(scheduled):
    f = scheduled
    response = f.dispatch()
    assert response['state'] == 'pending'
    assert f.queue.size() == 0
    run = f.dispatcher.get(response['check_run_id'])
    occurrence = f.schedules.list_executions(scheduled_task_id=f.task.scheduled_task_id)[0]
    assert occurrence.status == 'enqueued'
    assert run['effective_grants'][0]['grant_id'] == 'school-grant'
    assert run['analysis_checkpoint'] == run['acquisition_checkpoint'] == {}
    recovered = ProjectCheckDispatcher(ScheduledTaskStore(f.schedules.db_path), f.checks)
    worker = ProjectCheckWorker(recovered, adapter=one, clock=lambda: f.now)
    assert worker.run_once()['outcome'] == 'complete'
    assert worker.run_once() is None
    assert f.queue.size() == 0


@pytest.mark.parametrize('offset', [timedelta(hours=-1), timedelta(hours=1), timedelta(days=1), timedelta(days=7)])
def test_zero_off_window_provider_and_analysis_calls(scheduled, offset):
    f = scheduled
    run_id = f.dispatch()['check_run_id']
    f.now += offset
    worker = f.worker(lambda ctx: pytest.fail('off-window read or model call'))
    result = worker.run_once()
    assert result['outcome'] == 'partial'
    assert result['error_code'] == 'window_expired'
    assert f.dispatcher.get(run_id)['pages'] == 0


@pytest.mark.parametrize('operation', ['fetch', 'page', 'attachment', 'analysis', 'retry', 'continuation'])
def test_every_operation_stops_at_boundary_and_checkpoints_partial(scheduled, operation):
    f = scheduled
    f.dispatch()
    f.now += timedelta(minutes=59, seconds=59)
    calls = []
    def adapter(ctx):
        def read():
            calls.append(operation)
            f.now += timedelta(seconds=1)
            return CheckStep(acquisition_checkpoint={'page': 3}, analysis_checkpoint={'analyzed': 2})
        ctx.perform(operation, read)
        pytest.fail('continued after boundary')
    result = f.worker(adapter).run_once()
    assert calls == [operation]
    assert result['outcome'] == 'partial' and result['error_code'] == 'window_expired'
    assert result['acquisition_checkpoint'] == {'page': 3}
    assert result['analysis_checkpoint'] == {'analyzed': 2}
    assert f.worker(one).run_once() is None


def test_retry_cannot_cross_window_or_resume_on_tuesday(scheduled):
    f = scheduled
    f.dispatch()
    f.now += timedelta(minutes=59, seconds=58)
    def fail(ctx):
        return ctx.perform('page', lambda: (_ for _ in ()).throw(RuntimeError('secret')))
    result = f.worker(fail).run_once()
    assert result['outcome'] == 'partial' and result['error_code'] == 'retry_outside_window'
    assert 'secret' not in json.dumps(result)
    f.now += timedelta(days=1)
    assert f.worker(one).run_once() is None


def test_retry_backoff_and_pagination_budgets_survive_restart(scheduled):
    f = scheduled
    f.dispatch()
    def fail(ctx):
        return ctx.perform('page', lambda: (_ for _ in ()).throw(RuntimeError()))
    result = f.worker(fail).run_once()
    assert result['state'] == 'retry_wait' and result['pages'] == 1
    assert f.worker(one).run_once() is None
    f.now += timedelta(seconds=5)
    worker = f.worker(lambda ctx: ctx.perform('page', lambda: CheckStep(acquisition_checkpoint={'cursor': 'next'})))
    for _ in range(4):
        result = worker.run_once()
        assert result['acquisition_checkpoint'] == {'cursor': 'next'}
    result = f.worker(one).run_once()
    assert result['outcome'] == 'partial' and result['pages'] == 5


def test_claim_recovery_and_stale_checkpoint_rejected(scheduled):
    f = scheduled
    run_id = f.dispatch()['check_run_id']
    old = f.dispatcher.claim('dead-worker', f.now, 5)
    assert f.dispatcher.claim('other', f.now, 5) is None
    f.now += timedelta(seconds=6)
    new = f.dispatcher.claim('replacement', f.now, 30)
    assert new['check_run_id'] == run_id and new['generation'] == old['generation'] + 1
    with pytest.raises(CheckStopped, match='lease_lost'):
        f.dispatcher.save(old, now=f.now)
    assert f.dispatcher.get(run_id)['acquisition_checkpoint'] == {}


def test_concurrent_claims_one_winner(scheduled):
    f = scheduled
    f.dispatch()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda owner: f.dispatcher.claim(owner, f.now, 30), ['one', 'two']))
    assert sum(r is not None for r in results) == 1


@pytest.mark.parametrize('change', ['revision', 'grant', 'cancel', 'project'])
def test_recheck_authority_between_calls(scheduled, change):
    f = scheduled
    run_id = f.dispatch()['check_run_id']
    def adapter(ctx):
        ctx.perform('page', lambda: CheckStep(acquisition_checkpoint={'page': 1}))
        if change == 'revision':
            f.checks.set_status(AccountSession('child'), f.policy['policy_id'], expected_revision=1, status='paused')
        if change == 'grant':
            f.service.revoke_grant(AccountSession('parent'), grant_id='school-grant', expected_revision=1)
        if change == 'cancel':
            f.dispatcher.cancel(run_id, actor_user_id='child', now=f.now)
        if change == 'project':
            f.projects.set_status(f.project.project_id, 'paused', requester_user_id='child')
        return ctx.perform('attachment', lambda: pytest.fail('read after revocation'))
    result = f.worker(adapter).run_once()
    assert result['outcome'] == ('cancelled' if change == 'cancel' else 'blocked')
    assert result['acquisition_checkpoint'] == {'page': 1}


def test_missed_runs_coalesced_and_next_window_not_backlog(scheduled):
    f = scheduled
    f.now += timedelta(days=15)
    result = f.dispatch()
    run = f.dispatcher.get(result['check_run_id'])
    assert run['outcome'] == 'partial' and run['error_code'] == 'missed_window'
    assert f.schedules.get_task(f.task.scheduled_task_id).next_run_at > f.now.isoformat()
    assert f.scheduler.run_once(now=f.now) == []


def test_late_start_requires_explicit_bounded_policy(scheduled):
    f = scheduled
    f.now += timedelta(seconds=20)
    result = f.dispatcher.get(f.dispatch()['check_run_id'])
    assert result['error_code'] == 'missed_trigger'
    f.definition['missed_run'] = {'kind': 'bounded_late', 'max_lateness_seconds': 30}
    p = f.checks.save(AccountSession('child'), definition=f.definition, expected_revision=1,
        policy_id=f.policy['policy_id'], activate=True, authorization_reference='explicit-late')
    # Use a future Monday occurrence for this new authorized revision.
    f.dispatcher.sync(now=f.now)
    task = [t for t in f.schedules.list_tasks() if t.payload['policy_revision'] == 2][0]
    task = replace(task, next_run_at=(f.now - timedelta(seconds=20)).isoformat())
    f.schedules._save_task(task)
    response = f.scheduler.run_once(now=f.now)[0]
    assert f.dispatcher.get(response['check_run_id'])['state'] == 'pending'


def test_overlap_recorded_and_accounts_do_not_couple_policies(scheduled):
    f = scheduled
    first = f.dispatch()['check_run_id']
    # A separately authorized project policy can independently use the same grant.
    second_policy = active(f)
    f.dispatcher.sync(now=f.now)
    task = [t for t in f.schedules.list_tasks() if t.payload['policy_id'] == second_policy['policy_id']][0]
    f.schedules._save_task(replace(task, next_run_at=f.now.isoformat()))
    second = f.scheduler.run_once(now=f.now)[0]['check_run_id']
    assert first != second
    assert f.dispatcher.get(second)['state'] == 'pending'
    f.now += timedelta(days=7)
    response = f.scheduler.run_once(now=f.now)
    assert all(f.dispatcher.get(r['check_run_id'])['error_code'] == 'overlap' for r in response)


def test_end_date_and_missing_adapter_never_mean_no_match(scheduled):
    f = scheduled
    run_id = f.dispatch()['check_run_id']
    assert f.worker().run_once()['error_code'] == 'acquisition_adapter_unavailable'
    f.now += timedelta(days=41)
    responses = f.scheduler.run_once(now=f.now)
    assert all(f.dispatcher.get(r['check_run_id'])['outcome'] == 'blocked' for r in responses)
    assert f.checks._load(f.policy['policy_id'])['status'] == 'ended'


@pytest.mark.parametrize('zone,start,after,expected', [
    ('America/New_York', '2026-03-01T02:30:00', '2026-03-07T12:00:00+00:00', '2026-03-15T06:30:00+00:00'),
    ('America/New_York', '2026-10-25T01:30:00', '2026-10-31T12:00:00+00:00', '2026-11-01T05:30:00+00:00'),
    ('America/New_York', '2026-10-25T01:30:00', '2026-11-01T05:31:00+00:00', '2026-11-08T06:30:00+00:00'),
])
def test_dst_skip_gap_and_fire_fold_once(zone, start, after, expected):
    assert compute_next_run_at(schedule={'kind': 'rrule', 'dtstart': start, 'rrule': 'FREQ=WEEKLY'},
                               timezone_name=zone, after=datetime.fromisoformat(after)) == expected


@pytest.mark.parametrize('changes', [dict(job_kind='arbitrary'), dict(job_version=2),
    dict(payload={'policy_id': 'a', 'policy_revision': True}),
    dict(payload={'policy_id': 'a', 'policy_revision': 1, 'actor': 'forged'}), dict(prompt='poll')])
def test_typed_schema_strict(changes):
    args = dict(owner_user_id='child', name='Check', job_kind='project_check', payload={'policy_id': 'a', 'policy_revision': 1},
                schedule_kind='once', run_at='2026-01-01T00:00:00Z')
    args.update(changes)
    with pytest.raises(ValueError):
        ScheduledTaskStore().create_task(**args)


def test_dst_gap_is_durable_owner_scoped_status(tmp_path):
    store = ScheduledTaskStore(str(tmp_path / 'schedule.db'))
    task = store.create_task(owner_user_id='one', name='DST', prompt='Existing reminder',
        schedule_kind='rrule', rrule='FREQ=WEEKLY', dtstart='2026-03-01T02:30:00',
        timezone_name='America/New_York', now=datetime(2026, 3, 7, 12, tzinfo=timezone.utc))
    gaps = ScheduledTaskStore(store.db_path).list_schedule_gaps(task.scheduled_task_id, owner_user_id='one')
    assert len(gaps) == 1 and gaps[0]['reason'] == 'nonexistent_local_time'
    assert gaps[0]['local_time'].startswith('2026-03-08T02:30:00')
    with pytest.raises(PermissionError):
        store.list_schedule_gaps(task.scheduled_task_id, owner_user_id='other')


def test_stale_and_offline_receipts_are_owner_scoped(scheduled):
    f = scheduled
    f.dispatch()
    f.dispatcher.claim('dead-worker', f.now, 5)
    f.now += timedelta(days=1)
    session = AccountSession('child')
    result = f.dispatcher.status(session, f.policy['policy_id'], now=f.now)[0]
    assert result['offline'] and result['stale']
    with pytest.raises(PermissionError):
        f.dispatcher.status(AccountSession('other'), f.policy['policy_id'], now=f.now)
    with pytest.raises(PermissionError):
        f.dispatcher.cancel(result['check_run_id'], actor_user_id='other', now=f.now)
    assert f.worker(one).run_once()['error_code'] == 'window_expired'


def test_duplicate_dispatch_reconnects_to_same_logical_run(scheduled):
    f = scheduled
    occurrence = f.schedules.claim_due_occurrences(worker_id='first', now=f.now, lease_seconds=1)[0]
    accepted = f.dispatcher.dispatch(occurrence, now=f.now)
    f.now += timedelta(seconds=2)
    recovered = f.schedules.claim_expired_occurrences(worker_id='second', now=f.now)[0]
    assert recovered.scheduled_run_at == occurrence.scheduled_run_at
    assert f.dispatcher.dispatch(recovered, now=f.now)['check_run_id'] == accepted['check_run_id']


def test_unsupported_stored_version_does_not_fall_through_to_llm(scheduled):
    f = scheduled
    with f.schedules._connect() as conn:
        conn.execute('UPDATE v2_scheduled_tasks SET job_version=99')
    result = f.dispatch()
    assert result['error_code'] == 'unsupported_job_version'
    assert f.queue.size() == 0


def test_expired_lease_cannot_accept_inflight_result(scheduled):
    f = scheduled
    f.dispatch()
    def adapter(ctx):
        def slow():
            f.now += timedelta(seconds=31)
            return CheckStep(acquisition_checkpoint={'stale': True}, complete=True)
        return ctx.perform('page', slow)
    result = f.worker(adapter).run_once()
    assert result['state'] == 'leased' and result['acquisition_checkpoint'] == {}
    assert f.worker(one).run_once()['outcome'] == 'complete'


def test_in_memory_store_claims_are_safe_across_worker_threads():
    store = ScheduledTaskStore()
    now = datetime.now(timezone.utc)
    store.create_task(owner_user_id='one', name='One', prompt='Prompt', schedule_kind='once', run_at=now.isoformat())
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda owner: store.claim_due_occurrences(worker_id=owner, now=now), ['one', 'two']))
    assert sum(len(r) for r in results) == 1


def test_legacy_prompt_migration_preserves_ids_modes_and_recurrence(tmp_path):
    store = ScheduledTaskStore(str(tmp_path / 'legacy.db'))
    records = [store.create_task(owner_user_id='one', name=mode, prompt='Keep this exact prompt',
        delivery_mode=mode, schedule_kind='rrule', rrule='FREQ=WEEKLY',
        dtstart='2026-03-01T08:00:00Z') for mode in ['pdca', 'direct']]
    with store._connect() as conn:
        for column in ['job_kind', 'job_version', 'payload_json']:
            conn.execute(f'ALTER TABLE v2_scheduled_tasks DROP COLUMN {column}')
    migrated = ScheduledTaskStore(store.db_path)
    for record in records:
        restored = migrated.get_task(record.scheduled_task_id)
        assert restored == record
        assert restored.job_kind == 'prompt' and restored.job_version == 1


def test_byte_item_and_runtime_budgets_are_not_silently_complete(scheduled):
    f = scheduled
    f.dispatch()
    def adapter(ctx):
        def over_budget():
            f.now += timedelta(seconds=2)
            return CheckStep(acquisition_checkpoint={'page': 1}, items=101, complete=True)
        return ctx.perform('page', over_budget)
    result = f.worker(adapter).run_once()
    assert result['outcome'] == 'partial' and result['error_code'] == 'acquisition_budget'
    assert result['items'] == 101 and result['runtime_seconds'] == 2


def test_no_next_operation_after_runtime_deadline(scheduled):
    f = scheduled
    f.dispatch()
    def adapter(ctx):
        # Exceed runtime via separate short, leased steps.
        for page in range(4):
            def read():
                f.now += timedelta(seconds=20)
                return CheckStep(acquisition_checkpoint={'page': page})
            ctx.perform('page', read)
        pytest.fail('runtime overrun')
    result = f.worker(adapter).run_once()
    assert result['outcome'] == 'partial' and result['error_code'] == 'runtime_budget'
    assert result['pages'] == 3


def test_analysis_and_acquisition_advance_independently(scheduled):
    f = scheduled
    f.dispatch()
    def adapter(ctx):
        ctx.perform('page', lambda: CheckStep(acquisition_checkpoint={'cursor': 'done'}, items=100))
        return ctx.perform('analysis', lambda: CheckStep(analysis_checkpoint={'classified': 100}, complete=True))
    result = f.worker(adapter).run_once()
    assert result['outcome'] == 'complete'
    assert result['acquisition_checkpoint'] == {'cursor': 'done'}
    assert result['analysis_checkpoint'] == {'classified': 100}
