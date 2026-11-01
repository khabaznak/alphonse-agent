"""Durable provider-independent acquisition over the existing scheduled runs.

The scheduler database owns accepted pages and progress in one transaction.
Classification/CAPD checkpointing never controls provider cursor advancement.
Only references and coverage receipts enter the existing scoped event journal.
"""
from dataclasses import replace
from datetime import datetime, timedelta
import hashlib
import json

from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.automations import EventEnvelope
from alphonse.agent_v2.integrations.acquisition import (
    CursorExpired, ReadContext, ReadLimits, SourcePage,
)
from alphonse.agent_v2.integrations.content import normalize_mime, safe_link
from alphonse.agent_v2.services.project_check_worker import CheckStep, CheckStopped, instant

WORKER = 'native.source_acquisition'
EVENT = 'account.check_result'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class SourceAcquisition:
    def __init__(self, dispatcher, events, *, adapters=None, clock, triage_provider=None):
        self.dispatcher, self.events, self.clock = dispatcher, events, clock
        self.adapters = dict(adapters or {})
        with dispatcher.store._connect() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS v2_source_evidence (
                    evidence_id TEXT PRIMARY KEY, check_run_id TEXT NOT NULL,
                    scope_key TEXT NOT NULL, definition TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS v2_source_pages (
                    check_run_id TEXT NOT NULL, scope_key TEXT NOT NULL,
                    request_key TEXT NOT NULL, definition TEXT NOT NULL,
                    PRIMARY KEY(check_run_id,scope_key,request_key));
                CREATE TABLE IF NOT EXISTS v2_source_streams (
                    scope_key TEXT PRIMARY KEY, definition TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS v2_source_processing (
                    check_run_id TEXT NOT NULL, stage TEXT NOT NULL,
                    definition TEXT NOT NULL, PRIMARY KEY(check_run_id,stage));
                CREATE TABLE IF NOT EXISTS v2_source_publish_attempts (
                    check_run_id TEXT PRIMARY KEY, attempted_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS v2_source_publications (
                    check_run_id TEXT PRIMARY KEY, publication_id TEXT NOT NULL);
            ''')
        old_type = events.get_event_type(EVENT, '1')
        old_worker = events.get_worker(WORKER)
        events.register_event_type(event_type=EVENT, version='1', schema={
            'type': 'object', 'additionalProperties': False,
            'required': ['check_run_id', 'outcome', 'coverage'],
            'properties': {'check_run_id': {'type': 'string'}, 'outcome': {'type': 'string'},
                           'coverage': {'type': 'object'}}}, enabled=old_type.enabled if old_type else True)
        events.register_worker(worker_id=WORKER, display_name='Native source acquisition', allowed_event_types=[EVENT],
            enabled=old_worker.enabled if old_worker else True)
        events.bind_managed_adapter(WORKER, self)
        self.triage = None
        if triage_provider is not None:
            from alphonse.agent_v2.services.project_triage import ProjectTriage
            self.triage = ProjectTriage(self, triage_provider)

    def _policy(self, run):
        return self.dispatcher.policies.guard(run['policy_id'], expected_revision=run['policy_revision'], now=self.clock())

    def _context(self, ctx, policy, source):
        handle = self.dispatcher.policies.bindings.resolve_account(
            AccountSession(ctx.run['executing_user_id']), skill_id=source['skill_id'], name=source['role'],
            operation=source['operation'], resources=source['resources'],
            recipient_user_ids=policy['recipient_user_ids'], project_id=policy['project_id'],
            expected_binding_id=source['binding_id'], expected_connection_id=source['connection_id'],
            expected_grant_id=source['grant_id'], expected_grant_revision=source['grant_revision'])
        b, r = policy['budgets'], ctx.run
        return ReadContext(AccountSession(r['executing_user_id']), handle, ctx.deadline,
            ReadLimits(b['max_pages'] - r['pages'], b['max_items'] - r['items'], b['max_bytes'] - r['bytes_read']))

    def __call__(self, ctx):
        policy = ctx.guard()
        checkpoint = dict(ctx.run['acquisition_checkpoint'])
        index = checkpoint.get('source_index', 0)
        if index == len(policy['source_bindings']) and ctx.run['coverage'].get('acquisition_complete'):
            if ctx.run['coverage']['gaps']:
                raise CheckStopped('partial', 'incomplete_source_coverage')
            return self.triage(ctx) if self.triage else CheckStep(complete=True, no_match=ctx.run['items'] == 0)
        source = policy['source_bindings'][index]
        context = self._context(ctx, policy, source)
        adapter = self.adapters.get(context.grant.provider_key)
        if adapter is None:
            raise CheckStopped('blocked', 'acquisition_adapter_unavailable')
        # A new query, folder, owner, binding, grant scope or revision has a new
        # explicit bounded baseline. No cursor can silently cross that boundary.
        if not isinstance(adapter.version, str) or not adapter.version:
            raise ValueError('source_adapter_version_required')
        key = digest([policy['policy_id'], policy['executing_user_id'], source, policy['query_digest'], adapter.version])
        progress = checkpoint.get('progress') if checkpoint.get('scope_key') == key else None
        if progress is None:
            with self.dispatcher.store._connect() as conn:
                row = conn.execute('SELECT definition FROM v2_source_streams WHERE scope_key=?', (key,)).fetchone()
            progress = json.loads(row[0]) if row else dict(mode='baseline', cursor=None,
                since=(instant(ctx.run['scheduled_for']) - timedelta(days=policy['initial_lookback_days'])).isoformat())
        def fetch():
            nonlocal progress
            # This single transport operation receives its pre-reservation
            # allowance; perform durably charges the attempt before I/O.
            try:
                if progress['mode'] == 'changes':
                    page = adapter.changes(context, query=policy['query'], cursor=progress['cursor'])
                else:
                    page = adapter.search(context, query=policy['query'], since=instant(progress['since']), cursor=progress['cursor'])
            except CursorExpired:
                if progress.get('resynced'):
                    raise CheckStopped('partial', 'cursor_expired_again')
                progress = dict(mode='baseline', cursor=None, resynced=True,
                    since=(instant(ctx.run['scheduled_for']) - timedelta(days=policy['initial_lookback_days'])).isoformat())
                checkpoint.update(source_index=index, scope_key=key, progress=progress)
                ctx.run['coverage']['gaps'].append('cursor_expired_bounded_resync')
                return CheckStep(acquisition_checkpoint=checkpoint)
            return self._accept(ctx, policy, source, key, progress, page, checkpoint, index, context)
        return ctx.perform('page', fetch)

    def read_operation(self, ctx, *, source_index, operation, resource='', source_id='', attachment_id='', max_bytes=0, content_type=''):
        """Native provider tools use this gate for expansion/health/attachments.

        Raw attachment bytes are returned only to the trusted caller, never
        executed or opened. No linked URL or forwarded credential input exists.
        """
        policy = ctx.guard()
        source = policy['source_bindings'][source_index]
        context = self._context(ctx, policy, source)
        adapter = self.adapters.get(context.grant.provider_key)
        if adapter is None:
            raise CheckStopped('blocked', 'acquisition_adapter_unavailable')
        if operation not in {'health', 'capabilities', 'read', 'attachment'}:
            raise ValueError('source_operation_invalid')
        if operation in {'read', 'attachment'} and resource not in context.grant.resources:
            raise PermissionError('source_resource_denied')
        result = None
        def call():
            nonlocal result
            if operation in {'health', 'capabilities'}:
                result = getattr(adapter, operation)(context)
                size = len(json.dumps(result).encode())
                if size > context.limits.bytes:
                    raise CheckStopped('partial', 'source_content_budget')
                return CheckStep(bytes_read=size)
            if operation == 'attachment':
                if (content_type not in {'text/plain', 'text/csv', 'application/pdf'}
                        or type(max_bytes) is not int or not 0 < max_bytes <= min(context.limits.bytes, 5 * 1024 * 1024)):
                    raise ValueError('source_attachment_budget')
                result = adapter.attachment(context, resource=resource, source_id=source_id,
                                            attachment_id=attachment_id, max_bytes=max_bytes)
                if not isinstance(result, bytes) or len(result) > max_bytes:
                    raise CheckStopped('partial', 'source_attachment_budget')
                return CheckStep(items=1, bytes_read=len(result))
            result = adapter.read(context, resource=resource, source_id=source_id)
            if result.resource != resource or result.source_id != source_id:
                raise PermissionError('source_identity_mismatch')
            if len(result.mime) > context.limits.bytes:
                raise CheckStopped('partial', 'source_content_budget')
            # Expansion returns inert normalized evidence, never MIME execution.
            size = len(result.mime)
            result = normalize_mime(result.mime, max_bytes=context.limits.bytes)
            return CheckStep(items=1, bytes_read=size)
        ctx.perform('attachment' if operation == 'attachment' else 'fetch', call)
        return result

    def _accept(self, ctx, policy, source, key, progress, page, checkpoint, index, context):
        if (not isinstance(page, SourcePage) or type(page.bytes_read) is not int or page.bytes_read < 0
                or type(page.complete) is not bool or type(page.coverage_complete) is not bool
                or (page.cursor is not None and (not isinstance(page.cursor, str) or len(page.cursor) > 65536))
                or not page.complete and not page.cursor):
            raise ValueError('source_page_invalid')
        raw_bytes = sum(len(r.mime) for r in page.records)
        size = max(raw_bytes, page.bytes_read)
        if len(page.records) > context.limits.items or size > context.limits.bytes:
            raise CheckStopped('partial', 'source_page_budget')
        request = digest(progress)
        if not page.complete and page.cursor == progress['cursor']:
            raise CheckStopped('partial', 'source_cursor_stalled')
        records = []
        observed = self.clock().isoformat()
        for record in page.records:
            if (record.resource not in source['resources'] or not record.source_id or not record.version
                    or not record.source_timestamp):
                raise ValueError('source_identity_invalid')
            if datetime.fromisoformat(record.source_timestamp.replace('Z', '+00:00')).tzinfo is None:
                raise ValueError('source_timestamp_invalid')
            evidence_id = 'evidence_' + digest([ctx.run['check_run_id'], key, record.resource, record.source_id, record.version])
            records.append(dict(evidence_id=evidence_id, check_run_id=ctx.run['check_run_id'],
                source_identity=digest([source['connection_id'], record.resource, record.source_id, record.version]),
                owner_user_id=self.dispatcher.policies.bindings.integrations.get(source['connection_id']).connection['owner_user_id'],
                audience_user_ids=policy['recipient_user_ids'], content_digest=hashlib.sha256(record.mime).hexdigest(),
                state='deleted' if record.deleted else 'available',
                connection_id=source['connection_id'], grant_id=source['grant_id'], grant_revision=source['grant_revision'],
                actor_user_id=ctx.run['executing_user_id'], project_id=policy['project_id'],
                resource=record.resource, source_id=record.source_id, version=record.version,
                source_timestamp=record.source_timestamp, observed_at=observed,
                safe_url=safe_link(record.safe_url), deleted=record.deleted,
                content={} if record.deleted else normalize_mime(record.mime, max_bytes=context.limits.bytes),
                retention_deadline=(self.clock() + timedelta(days=7)).isoformat()))
        size = max(size, sum(len(json.dumps(r['content']).encode()) for r in records))
        if size > context.limits.bytes:
            raise CheckStopped('partial', 'source_content_budget')
        next_progress = dict(progress, cursor=page.cursor)
        if page.complete:
            # No incremental token means the next occurrence repeats a bounded
            # baseline, never treats an absent cursor as full change coverage.
            next_progress = dict(mode='changes', cursor=page.cursor) if page.cursor else dict(
                mode='baseline', cursor=None, since=progress.get('since') or
                (instant(ctx.run['scheduled_for']) - timedelta(days=policy['initial_lookback_days'])).isoformat())
            checkpoint = {'source_index': index + 1}
        else:
            checkpoint = dict(source_index=index, scope_key=key, progress=next_progress)
        done = page.complete and index + 1 == len(policy['source_bindings'])
        if not page.coverage_complete:
            ctx.run['coverage']['gaps'].append('provider_incomplete_coverage')
        coverage = dict(ctx.run['coverage'], acquisition_complete=done,
            processing='pending' if ctx.run['items'] + len(records) else 'not_required')
        coverage['sources'] = dict(coverage.get('sources', {}))
        coverage['sources'][key] = dict(resources=source['resources'], query_digest=policy['query_digest'],
            baseline_since=progress.get('since'), complete=page.complete,
            coverage_complete=page.coverage_complete, observed_at=observed)
        receipt = {k: v for k, v in ctx.run.items() if k not in {
            'state', 'lease_owner', 'generation', 'lease_until', 'next_attempt_at'}}
        receipt.update(acquisition_checkpoint=checkpoint, coverage=coverage,
                       items=ctx.run['items'] + len(records), bytes_read=ctx.run['bytes_read'] + size)
        # Fence the transaction without calling guard: in-flight bytes arriving
        # at a window boundary may be stored, but cannot cause another read.
        with self.dispatcher.store._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            updated = conn.execute("""UPDATE v2_project_check_runs SET receipt=? WHERE check_run_id=?
                AND state='leased' AND generation=? AND lease_owner=? AND lease_until>?""",
                (json.dumps(receipt), ctx.run['check_run_id'], ctx.run['generation'], ctx.run['lease_owner'], observed))
            if updated.rowcount != 1:
                raise CheckStopped('blocked', 'lease_lost')
            conn.execute('INSERT INTO v2_source_pages VALUES(?,?,?,?)',
                (ctx.run['check_run_id'], key, request, json.dumps(dict(evidence_ids=[r['evidence_id'] for r in records],
                complete=page.complete, coverage_complete=page.coverage_complete, observed_at=observed))))
            for record in records:
                if record['deleted']:
                    old_rows = conn.execute('SELECT evidence_id,definition FROM v2_source_evidence').fetchall()
                    for old_row in old_rows:
                        old = json.loads(old_row['definition'])
                        if (old['connection_id'], old['resource'], old['source_id']) == (record['connection_id'], record['resource'], record['source_id']):
                            old.update(content={}, state='deleted')
                            conn.execute('UPDATE v2_source_evidence SET definition=? WHERE evidence_id=?', (json.dumps(old), old_row['evidence_id']))
                conn.execute('INSERT OR IGNORE INTO v2_source_evidence VALUES(?,?,?,?)',
                             (record['evidence_id'], ctx.run['check_run_id'], key, json.dumps(record)))
            conn.execute('INSERT OR REPLACE INTO v2_source_streams VALUES(?,?)', (key, json.dumps(next_progress)))
        # The evidence transaction already accounts for bytes/items; perform
        # must not add them again. A crash now recovers the persisted receipt.
        ctx.run.update(acquisition_checkpoint=checkpoint, coverage=coverage,
                       items=receipt['items'], bytes_read=receipt['bytes_read'])
        if done and coverage['gaps']:
            raise CheckStopped('partial', 'incomplete_source_coverage')
        return CheckStep(acquisition_checkpoint=checkpoint, complete=done and self.triage is None,
                         no_match=done and ctx.run['items'] == 0 and self.triage is None)

    def evidence(self, session, run_id):
        run = self.dispatcher.get(run_id)
        if session.actor_user_id != run['executing_user_id'] or run['outcome'] == 'cancelled':
            raise PermissionError('source_evidence_denied')
        if not instant(run['window_start']) <= self.clock() < instant(run['window_end']):
            raise PermissionError('source_evidence_outside_window')
        self._policy(run)  # Revalidate current grants, policy, project and actor.
        with self.dispatcher.store._connect() as conn:
            rows = conn.execute('SELECT definition FROM v2_source_evidence WHERE check_run_id=?', (run_id,)).fetchall()
        return [r for row in rows if (r := json.loads(row[0]))['state'] == 'available'
                and instant(r['retention_deadline']) > self.clock()]

    def processing(self, session, run_id):
        self.evidence(session, run_id)  # Current scope, lifecycle and window gate.
        with self.dispatcher.store._connect() as conn:
            rows = conn.execute('SELECT stage,definition FROM v2_source_processing WHERE check_run_id=?', (run_id,)).fetchall()
        return {row['stage']: json.loads(row['definition']) for row in rows}

    def checkpoint_processing(self, session, run_id, *, stage, evidence_ids):
        if stage not in {'classification', 'capd'}:
            raise ValueError('source_processing_stage_invalid')
        run = self.dispatcher.get(run_id)
        # Analysis remains window-bound even though source acquisition is done.
        if not instant(run['window_start']) <= self.clock() < instant(run['window_end']):
            raise PermissionError('source_analysis_outside_window')
        allowed = {r['evidence_id'] for r in self.evidence(session, run_id)}
        if not set(evidence_ids) <= allowed:
            raise PermissionError('source_evidence_denied')
        with self.dispatcher.store._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            old = conn.execute('SELECT definition FROM v2_source_processing WHERE check_run_id=? AND stage=?', (run_id, stage)).fetchone()
            completed = set(json.loads(old[0])) if old else set()
            conn.execute('INSERT OR REPLACE INTO v2_source_processing VALUES(?,?,?)',
                         (run_id, stage, json.dumps(sorted(completed | set(evidence_ids)))))

    def resolve(self, worker_id, check_run_id):
        run = self.dispatcher.get(check_run_id)
        return EventEnvelope(publication_id='source-result:' + check_run_id, worker_id=WORKER,
            actor_user_id=run['executing_user_id'], owner_user_id=run['executing_user_id'],
            project_id=run['project_id'], policy_id=run['policy_id'], policy_revision=run['policy_revision'],
            check_run_id=check_run_id, resource_id=run['policy_id'], event_type=EVENT, event_version='1',
            occurred_at=run['scheduled_for'], observed_at=run['finished_at'] or run['scheduled_for'])

    def authorize(self, envelope, *, actor_user_id, project_id, phase):
        run = self.dispatcher.get(envelope.check_run_id)
        self._policy(run)
        if phase in {'dispatch', 'expansion'} and not instant(run['window_start']) <= self.clock() < instant(run['window_end']):
            return False
        expected = self.resolve(WORKER, envelope.check_run_id)
        if phase in {'subscription', 'catalog'}:
            # A subscription may be installed before the first result; its
            # stored scope remains valid when the observed timestamp is sealed.
            expected = replace(expected, observed_at=envelope.observed_at)
        elif run['state'] != 'terminal':
            return False
        return (envelope == expected and actor_user_id == run['executing_user_id']
                and project_id == run['project_id'] and run['outcome'] != 'cancelled')

    def purge_content(self):
        """Remove expired/revoked bytes; keep content-free provenance receipts."""
        with self.dispatcher.store._connect() as conn:
            rows = conn.execute("SELECT evidence_id,check_run_id,definition FROM v2_source_evidence WHERE json_extract(definition, '$.state') = 'available'").fetchall()
        for row in rows:
            value = json.loads(row['definition'])
            state = 'expired' if instant(value['retention_deadline']) <= self.clock() else ''
            if not state:
                try:
                    run = self.dispatcher.get(row['check_run_id'])
                    if run['outcome'] == 'cancelled':
                        raise PermissionError('cancelled')
                    self._policy(run)
                except (PermissionError, ValueError, KeyError, OSError):
                    state = 'revoked'
            if state:
                value.update(content={}, state=state)
                with self.dispatcher.store._connect() as conn:
                    conn.execute('UPDATE v2_source_evidence SET definition=? WHERE evidence_id=?',
                                 (json.dumps(value), row['evidence_id']))

    def publish_ready(self):
        """Crash-safe retry via existing durable event receipts; no second queue."""
        self.purge_content()
        with self.dispatcher.store._connect() as conn:
            rows = conn.execute("""SELECT r.check_run_id FROM v2_project_check_runs r
                LEFT JOIN v2_source_publish_attempts a ON a.check_run_id=r.check_run_id
                WHERE state='terminal' AND json_extract(receipt, '$.outcome') NOT IN ('cancelled','no_match')
                AND NOT (json_extract(receipt, '$.outcome')='complete' AND COALESCE(json_extract(receipt, '$.coverage.triage_no_match'),0)=1)
                AND r.check_run_id NOT IN
                (SELECT check_run_id FROM v2_source_publications) ORDER BY COALESCE(a.attempted_at,''),r.rowid LIMIT 100""").fetchall()
        for row in rows:
            run = self.dispatcher.get(row[0])
            if run['outcome'] in {'cancelled', 'no_match'}:
                continue
            with self.dispatcher.store._connect() as conn:
                conn.execute('INSERT OR REPLACE INTO v2_source_publish_attempts VALUES(?,?)', (row[0], self.clock().isoformat()))
            response = self.events.publish_managed(worker_id=WORKER, check_run_id=row[0],
                event_id='source-result:' + row[0], event_type=EVENT, event_version='1',
                occurred_at=run['scheduled_for'], payload=dict(check_run_id=row[0],
                outcome=run['outcome'], coverage=run['coverage']))
            if response.get('accepted'):
                with self.dispatcher.store._connect() as conn:
                    conn.execute('INSERT OR IGNORE INTO v2_source_publications VALUES(?,?)',
                                 (row[0], 'source-result:' + row[0]))
                    # This is diagnostic only; event receipts are the dedupe authority.
                    stored = conn.execute('SELECT receipt FROM v2_project_check_runs WHERE check_run_id=?', (row[0],)).fetchone()
                    receipt = json.loads(stored[0])
                    receipt['publication_id'] = 'source-result:' + row[0]
                    conn.execute('UPDATE v2_project_check_runs SET receipt=? WHERE check_run_id=?', (json.dumps(receipt), row[0]))
