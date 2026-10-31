"""Deterministic project-check dispatch and leased, window-bound acquisition.

Adapters are trusted daemon code (#114), never model-authored jobs. Each adapter
step must use context.perform for *every* provider read, attachment, retry and
analysis call. A step result is checkpointed before its post-call fence, so an
in-flight call finishing at the boundary leaves truthful partial progress.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
import hashlib
import json
import threading
from typing import Any, Callable
from uuid import uuid4
from zoneinfo import ZoneInfo

from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore, ScheduledOccurrence
from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.project_checks import ProjectCheckService


def instant(value: str) -> datetime:
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)


def window_for(policy: dict, scheduled: datetime) -> tuple[datetime, datetime]:
    zone = ZoneInfo(policy['timezone'])
    day = scheduled.astimezone(zone).date()
    window = policy['retrieval_window']
    start, end = [datetime.combine(day, time.fromisoformat(window[k]), zone).astimezone(timezone.utc)
                  for k in ('start', 'end')]
    start = max(start, instant(policy['duration']['starts_at']))
    if policy['duration']['ends_at']:
        end = min(end, instant(policy['duration']['ends_at']))
    return start, end


class CheckStopped(Exception):
    def __init__(self, outcome: str, code: str):
        self.outcome, self.code = outcome, code
        super().__init__(code)


@dataclass(frozen=True)
class CheckStep:
    """Provider evidence persistence belongs to the adapter; cursors stay separate.

    Only return complete once all required analysis has durably finished. An
    unavailable adapter, an empty error response or a budget stop is never no_match.
    """
    acquisition_checkpoint: dict | None = None
    analysis_checkpoint: dict | None = None
    items: int = 0
    bytes_read: int = 0
    complete: bool = False


class ProjectCheckDispatcher:
    def __init__(self, store: ScheduledTaskStore, policies: ProjectCheckService):
        self.store, self.policies = store, policies
        with store._connect() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS v2_project_check_runs (
                    check_run_id TEXT PRIMARY KEY, policy_id TEXT NOT NULL,
                    policy_revision INTEGER NOT NULL, occurrence_key TEXT NOT NULL UNIQUE,
                    state TEXT NOT NULL, lease_owner TEXT NOT NULL DEFAULT '',
                    generation INTEGER NOT NULL DEFAULT 0, lease_until TEXT NOT NULL DEFAULT '',
                    next_attempt_at TEXT NOT NULL DEFAULT '', receipt TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_check_runs_ready ON v2_project_check_runs(state, next_attempt_at);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_check_policy_inflight ON v2_project_check_runs(policy_id)
                    WHERE state != 'terminal';
            ''')

    def sync(self, *, now: datetime) -> None:
        """Project authority remains in its store; schedule installation grants nothing.

        Revision-specific immutable payloads and guard-on-dispatch make interruption
        between stores fail closed. No existing payload ever points at a new scope.
        """
        with self.policies.projects._connect() as conn:
            policies = [json.loads(r[0]) for r in conn.execute('SELECT definition FROM v2_project_check_policies')]
        revisions = {p['policy_id']: p for p in policies}
        with self.store._connect() as conn:
            tasks = conn.execute("SELECT scheduled_task_id, payload_json FROM v2_scheduled_tasks WHERE job_kind='project_check' AND status='active'").fetchall()
        for task in tasks:
            payload = json.loads(task['payload_json'])
            policy = revisions.get(payload.get('policy_id'))
            if not policy or policy['status'] != 'active' or policy['revision'] != payload.get('policy_revision'):
                self.store.cancel_task(task['scheduled_task_id'])
        for p in policies:
            if p['status'] != 'active':
                continue
            trigger = p['trigger']
            hour, minute = trigger['local_time'].split(':')
            weekdays = ','.join(('MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU')[d] for d in trigger['weekdays'])
            anchor = max(instant(p['duration']['starts_at']), instant(p['authorized_at']))
            self.store.create_task(owner_user_id=p['executing_user_id'], project_id=p['project_id'],
                name='Project check', job_kind='project_check',
                payload={'policy_id': p['policy_id'], 'policy_revision': p['revision']},
                schedule_kind='rrule', timezone_name=p['timezone'],
                rrule=f'FREQ=WEEKLY;BYDAY={weekdays};BYHOUR={hour};BYMINUTE={minute};BYSECOND=0',
                dtstart=anchor.isoformat(), now=anchor - timedelta(microseconds=1),
                idempotency_key=f"project_check:{p['policy_id']}:{p['revision']}")

    def dispatch(self, occurrence: ScheduledOccurrence, *, now: datetime) -> dict:
        task = occurrence.task
        payload = task.payload
        code, outcome = '', ''
        policy = self.policies._load(payload['policy_id'])
        start, end = window_for(policy, instant(occurrence.scheduled_run_at))
        # The current policy's window is not authority for an old payload.
        try:
            self.policies.guard(payload['policy_id'], expected_revision=payload['policy_revision'], now=now)
            local = now.astimezone(ZoneInfo(policy['timezone']))
            if not start <= now < end or local.weekday() not in policy['retrieval_window']['weekdays']:
                raise CheckStopped('partial', 'missed_window')
            missed = policy['missed_run']
            # Late dispatch is limited to the original authorized retrieval window.
            # next_allowed_window skips late starts; normal polling has a 1s tolerance.
            lateness = (now - instant(occurrence.scheduled_run_at)).total_seconds()
            tolerance = missed.get('max_lateness_seconds', 1)
            if lateness > tolerance:
                raise CheckStopped('partial', 'missed_trigger')
        except CheckStopped as exc:
            outcome, code = exc.outcome, exc.code
        except (PermissionError, ValueError, KeyError, OSError):
            outcome, code = 'blocked', 'policy_unavailable'
        run_id = 'check_run_' + hashlib.sha256(occurrence.occurrence_key.encode()).hexdigest()[:32]
        receipt = dict(check_run_id=run_id, policy_id=payload['policy_id'], policy_revision=payload['policy_revision'],
            occurrence_key=occurrence.occurrence_key, scheduled_for=occurrence.scheduled_run_at,
            executing_user_id=policy['executing_user_id'], project_id=policy['project_id'],
            effective_grants=[{k: s[k] for k in ('grant_id', 'grant_revision', 'connection_id')} for s in policy['source_bindings']],
            query_digest=policy['query_digest'], window_start=start.isoformat(), window_end=end.isoformat(),
            acquisition_checkpoint={}, analysis_checkpoint={}, pages=0, items=0, bytes_read=0, model_calls=0,
            attempts=0, runtime_seconds=0, started_at='', finished_at=now.isoformat() if outcome else '',
            outcome=outcome, error_code=code, coverage={'gaps': [code] if code else []}, publication_id='')
        if code in {'missed_window', 'missed_trigger'}:
            receipt['coverage'].update(missed_from=occurrence.scheduled_run_at, missed_until=now.isoformat())
        with self.store._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            existing = conn.execute('SELECT check_run_id, state FROM v2_project_check_runs WHERE occurrence_key=?', (occurrence.occurrence_key,)).fetchone()
            if existing:
                return dict(existing)
            overlap = conn.execute("SELECT check_run_id FROM v2_project_check_runs WHERE policy_id=? AND state!='terminal'", (payload['policy_id'],)).fetchone()
            if overlap and not outcome:
                receipt.update(outcome='partial', error_code='overlap', finished_at=now.isoformat(),
                               coverage={'gaps': ['overlap'], 'overlapping_run_id': overlap[0]})
                outcome = 'partial'
            state = 'terminal' if outcome else 'pending'
            conn.execute('INSERT INTO v2_project_check_runs(check_run_id,policy_id,policy_revision,occurrence_key,state,receipt) VALUES(?,?,?,?,?,?)',
                         (run_id, payload['policy_id'], payload['policy_revision'], occurrence.occurrence_key, state, json.dumps(receipt)))
        return dict(check_run_id=run_id, state=state, error_code=receipt['error_code'])

    def get(self, run_id: str) -> dict:
        with self.store._connect() as conn:
            row = conn.execute('SELECT * FROM v2_project_check_runs WHERE check_run_id=?', (run_id,)).fetchone()
        if not row:
            raise KeyError(run_id)
        result = dict(row)
        result.update(json.loads(result.pop('receipt')))
        return result

    def status(self, session: AccountSession, policy_id: str, *, now: datetime, limit: int = 100) -> list[dict]:
        """Authenticated operational receipts; no global private-account catalog."""
        self.policies._managed(session, policy_id)
        with self.store._connect() as conn:
            ids = [r[0] for r in conn.execute(
                'SELECT check_run_id FROM v2_project_check_runs WHERE policy_id=? ORDER BY rowid DESC LIMIT ?',
                (policy_id, max(1, min(limit, 100))))]
        result = []
        for run_id in ids:
            run = self.get(run_id)
            run['stale'] = run['state'] != 'terminal' and now >= instant(run['window_end'])
            run['offline'] = run['state'] == 'leased' and now >= instant(run['lease_until'])
            result.append(run)
        return result

    def heartbeat(self, run: dict, *, now: datetime, lease_seconds: float) -> bool:
        with self.store._connect() as conn:
            return conn.execute("""UPDATE v2_project_check_runs SET lease_until=?
                WHERE check_run_id=? AND state='leased' AND generation=? AND lease_owner=? AND lease_until>?""",
                ((now + timedelta(seconds=lease_seconds)).isoformat(), run['check_run_id'],
                 run['generation'], run['lease_owner'], now.isoformat())).rowcount == 1

    def cancel(self, run_id: str, *, actor_user_id: str, now: datetime) -> None:
        run = self.get(run_id)
        if run['executing_user_id'] != actor_user_id:
            raise PermissionError('project_check_owner_required')
        with self.store._connect() as conn:
            row = conn.execute('SELECT receipt FROM v2_project_check_runs WHERE check_run_id=?', (run_id,)).fetchone()
            receipt = json.loads(row[0])
            receipt.update(outcome='cancelled', error_code='cancelled', finished_at=now.isoformat())
            conn.execute("UPDATE v2_project_check_runs SET state='terminal', generation=generation+1, receipt=? WHERE check_run_id=? AND state!='terminal'", (json.dumps(receipt), run_id))

    def claim(self, owner: str, now: datetime, lease_seconds: float) -> dict | None:
        with self.store._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute("""SELECT * FROM v2_project_check_runs
                WHERE state='pending' OR (state='retry_wait' AND next_attempt_at<=?)
                    OR (state='leased' AND lease_until<=?)
                ORDER BY rowid LIMIT 1""", (now.isoformat(), now.isoformat())).fetchone()
            if not row:
                return None
            receipt = json.loads(row['receipt'])
            receipt['attempts'] += 1
            receipt['started_at'] = receipt['started_at'] or now.isoformat()
            until = (now + timedelta(seconds=lease_seconds)).isoformat()
            generation = row['generation'] + 1
            conn.execute("UPDATE v2_project_check_runs SET state='leased',lease_owner=?,generation=?,lease_until=?,receipt=? WHERE check_run_id=?",
                         (owner, generation, until, json.dumps(receipt), row['check_run_id']))
        return self.get(row['check_run_id'])

    def save(self, run: dict, *, now: datetime, state: str = 'leased', lease_seconds: float = 30) -> None:
        receipt = {k: v for k, v in run.items() if k not in {'state', 'lease_owner', 'generation', 'lease_until', 'next_attempt_at'}}
        with self.store._connect() as conn:
            cursor = conn.execute("""UPDATE v2_project_check_runs SET state=?,receipt=?,lease_until=?,next_attempt_at=?
                WHERE check_run_id=? AND state='leased' AND lease_owner=? AND generation=? AND lease_until>?""",
                (state, json.dumps(receipt), (now + timedelta(seconds=lease_seconds)).isoformat(),
                 run.get('next_attempt_at', ''), run['check_run_id'], run['lease_owner'], run['generation'], now.isoformat()))
            if cursor.rowcount != 1:
                raise CheckStopped('blocked', 'lease_lost')


class CheckContext:
    def __init__(self, worker: 'ProjectCheckWorker', run: dict):
        self.worker, self.run = worker, run
        self.operation_count = 0

    def guard(self) -> dict:
        worker, run = self.worker, self.run
        now = worker.clock()
        latest = worker.dispatcher.get(run['check_run_id'])
        if latest['outcome'] == 'cancelled' or worker._stop.is_set():
            raise CheckStopped('cancelled', 'cancelled')
        if latest['generation'] != run['generation'] or instant(latest['lease_until']) <= now:
            raise CheckStopped('blocked', 'lease_lost')
        if not instant(run['window_start']) <= now < instant(run['window_end']):
            raise CheckStopped('partial', 'window_expired')
        try:
            policy = worker.dispatcher.policies.guard(run['policy_id'], expected_revision=run['policy_revision'], now=now)
        except (PermissionError, ValueError, KeyError, OSError) as exc:
            raise CheckStopped('blocked', 'policy_unavailable') from exc
        local = now.astimezone(ZoneInfo(policy['timezone']))
        window = policy['retrieval_window']
        if local.weekday() not in window['weekdays'] or not window['start'] <= local.strftime('%H:%M') < window['end']:
            raise CheckStopped('partial', 'window_expired')
        budget = policy['budgets']
        if run['runtime_seconds'] >= budget['max_runtime_seconds']:
            raise CheckStopped('partial', 'runtime_budget')
        worker.dispatcher.save(run, now=now, lease_seconds=worker.lease_seconds)
        return policy

    def perform(self, operation: str, call: Callable[[], CheckStep]) -> CheckStep:
        """Exactly one bounded operation; adapters must disable implicit retries.

        The context deadline and remaining byte/item budgets must be passed to the
        provider transport. In-flight I/O cannot be recalled; its next operation is
        fenced. A call exceeding its lease cannot commit any checkpoint.
        """
        policy = self.guard()
        if operation not in {'fetch', 'page', 'attachment', 'analysis', 'continuation', 'retry'}:
            raise ValueError('project_check_operation_invalid')
        run, budget = self.run, policy['budgets']
        metric = 'model_calls' if operation == 'analysis' else 'pages'
        maximum = budget['max_model_calls'] if metric == 'model_calls' else budget['max_pages']
        if run[metric] >= maximum or (operation != 'analysis' and (
                run['items'] >= budget['max_items'] or run['bytes_read'] >= budget['max_bytes'])):
            raise CheckStopped('partial', 'acquisition_budget')
        # Reserve before I/O, including failed calls, so retries cannot reset limits.
        run[metric] += 1
        self.worker.dispatcher.save(run, now=self.worker.clock(), lease_seconds=self.worker.lease_seconds)
        started = self.worker.clock()
        finished = threading.Event()
        deadline = min(instant(run['window_end']), started + timedelta(
            seconds=budget['max_runtime_seconds'] - run['runtime_seconds']))
        def heartbeat():
            while not finished.wait(max(0.01, self.worker.lease_seconds / 3)):
                now = self.worker.clock()
                if self.worker._stop.is_set() or now >= deadline:
                    return
                try:
                    if not self.worker.dispatcher.heartbeat(run, now=now, lease_seconds=self.worker.lease_seconds):
                        return
                except Exception:
                    return  # No renewal is safer than extending uncertain ownership.
        pulse = threading.Thread(target=heartbeat, name='alphonse-check-lease', daemon=True)
        pulse.start()
        try:
            result = call()
        finally:
            finished.set()
            pulse.join(timeout=1)
            run['runtime_seconds'] += max(0, (self.worker.clock() - started).total_seconds())
            self.worker.dispatcher.save(run, now=self.worker.clock(), lease_seconds=self.worker.lease_seconds)
        if (not isinstance(result, CheckStep) or type(result.items) is not int or result.items < 0
                or type(result.bytes_read) is not int or result.bytes_read < 0
                or any(c is not None and not isinstance(c, dict) for c in
                       (result.acquisition_checkpoint, result.analysis_checkpoint))):
            raise ValueError('project_check_step_invalid')
        for name in ('acquisition_checkpoint', 'analysis_checkpoint'):
            checkpoint = getattr(result, name)
            if checkpoint is not None:
                run[name] = checkpoint
        run.update(items=run['items'] + result.items, bytes_read=run['bytes_read'] + result.bytes_read)
        self.worker.dispatcher.save(run, now=self.worker.clock(), lease_seconds=self.worker.lease_seconds)
        self.operation_count += 1
        self.guard()
        if run['items'] > budget['max_items'] or run['bytes_read'] > budget['max_bytes']:
            raise CheckStopped('partial', 'acquisition_budget')
        return result

    @property
    def deadline(self) -> datetime:
        policy = self.guard()
        return min(instant(self.run['window_end']), self.worker.clock() + timedelta(
            seconds=policy['budgets']['max_runtime_seconds'] - self.run['runtime_seconds']))


class ProjectCheckWorker:
    """One bounded step at a time, separate from the scheduler's dispatch thread."""
    def __init__(self, dispatcher: ProjectCheckDispatcher, *, adapter: Callable[[CheckContext], CheckStep] | None = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc), lease_seconds: float = 30,
                 max_attempts: int = 5):
        self.dispatcher, self.adapter, self.clock = dispatcher, adapter, clock
        self.lease_seconds, self.max_attempts = lease_seconds, max_attempts
        self.owner = 'acquisition-' + uuid4().hex
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error = ''

    def run_once(self) -> dict | None:
        run = self.dispatcher.claim(self.owner, self.clock(), self.lease_seconds)
        if run is None:
            return None
        context = CheckContext(self, run)
        state = 'terminal'
        try:
            context.guard()
            if run['attempts'] > self.max_attempts:
                raise CheckStopped('failed', 'attempts_exhausted')
            if self.adapter is None:
                raise CheckStopped('blocked', 'acquisition_adapter_unavailable')
            result = self.adapter(context)
            if not context.operation_count:
                raise CheckStopped('failed', 'unguarded_adapter_step')
            context.guard()
            if result.complete:
                run.update(outcome='complete', error_code='')
            else:
                state = 'pending'
                # Continuations do not consume retry attempts.
                run['attempts'] -= 1
        except CheckStopped as exc:
            if exc.code == 'lease_lost':
                return self.dispatcher.get(run['check_run_id'])
            run.update(outcome=exc.outcome, error_code=exc.code)
        except Exception:
            # Do not persist provider exception strings (may contain private data).
            if run['attempts'] >= self.max_attempts:
                run.update(outcome='failed', error_code='acquisition_failed')
            else:
                retry = self.clock() + timedelta(seconds=min(3600, 5 * 2 ** (run['attempts'] - 1)))
                if retry >= instant(run['window_end']):
                    run.update(outcome='partial', error_code='retry_outside_window')
                else:
                    run['next_attempt_at'] = retry.isoformat()
                    run['error_code'] = 'acquisition_retry'
                    state = 'retry_wait'
        if state == 'terminal':
            run['finished_at'] = self.clock().isoformat()
            if run['outcome'] != 'complete':
                run['coverage']['gaps'].append(run['error_code'])
        try:
            self.dispatcher.save(run, now=self.clock(), state=state, lease_seconds=self.lease_seconds)
        except CheckStopped:
            pass  # Cancellation/new lease won; its receipt is authoritative.
        return self.dispatcher.get(run['check_run_id'])

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name='alphonse-project-acquisition', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
                self.last_error = ''
            except Exception:
                self.last_error = 'project_check_worker_error'
            self._stop.wait(0.1)
