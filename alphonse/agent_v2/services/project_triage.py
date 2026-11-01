"""Durable bounded System One stage over acquired, currently authorized evidence."""
import json

from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.project_triage import VERSION, CONTRACT_DIGEST, MAX_CANDIDATES, MAX_CONTENT_BYTES
from alphonse.agent_v2.services.project_check_worker import CheckStep, CheckStopped


class ProjectTriage:
    def __init__(self, acquisition, provider):
        self.acquisition, self.provider = acquisition, provider
        with acquisition.dispatcher.store._connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS v2_project_triage (
                check_run_id TEXT NOT NULL, evidence_id TEXT NOT NULL,
                definition TEXT NOT NULL, PRIMARY KEY(check_run_id,evidence_id))''')

    def receipts(self, session, run_id):
        self.acquisition.evidence(session, run_id)
        with self.acquisition.dispatcher.store._connect() as conn:
            rows = conn.execute('SELECT definition FROM v2_project_triage WHERE check_run_id=?', (run_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def __call__(self, ctx):
        policy = ctx.guard()
        run_id = ctx.run['check_run_id']
        session = AccountSession(ctx.run['executing_user_id'])
        records = self.acquisition.evidence(session, run_id)
        receipts = {r['evidence_id']: r for r in self.receipts(session, run_id)}
        for record in records:
            old = receipts.get(record['evidence_id'])
            if old and (old['content_digest'] != record['content_digest'] or old['source_version'] != record['version']):
                raise CheckStopped('blocked', 'project_triage_evidence_changed')
        # Recover a crash between durable decision receipts and the independent
        # processing checkpoint without another provider/model request.
        self.acquisition.checkpoint_processing(session, run_id, stage='classification', evidence_ids=[
            r['evidence_id'] for r in records if r['evidence_id'] in receipts
            and receipts[r['evidence_id']]['status'] != 'pending'])
        remaining = [r for r in records if r['evidence_id'] not in receipts]
        if not remaining:
            pending = any(r['status'] == 'pending' for r in receipts.values())
            if pending:
                raise CheckStopped('partial', 'project_triage_pending')
            ctx.run['coverage']['triage_no_match'] = bool(records) and all(
                r['status'] == 'irrelevant' for r in receipts.values()) and not ctx.run['coverage']['gaps']
            return CheckStep(complete=True, no_match=not records)
        batch = remaining[:MAX_CANDIDATES]
        candidates, eligible = [], []
        outcomes = {r['evidence_id']: 'pending' for r in batch}
        telemetry = dict(version=VERSION, contract_digest=CONTRACT_DIGEST, questions=0, usage={}, status='content_budget')
        for record in batch:
            content = record['content']
            if len(json.dumps(content).encode()) > MAX_CONTENT_BYTES or content.get('attachments'):
                continue  # Never clip unseen evidence into a confident negative.
            eligible.append(record)
            candidates.append(dict(evidence_id=record['evidence_id'], content=content,
                source_timestamp=record['source_timestamp'], observed_at=record['observed_at']))
        try:
            provider = self.provider()
        except Exception:
            provider = None
        if eligible and provider is not None:
            def call():
                nonlocal telemetry
                results, telemetry = provider.triage_project_check(
                    goal=dict(query=policy['query'], project=self.acquisition.dispatcher.policies._snapshot(
                        session, policy['project_id'])), candidates=candidates,
                    timeout_seconds=max(1, min(10, (ctx.deadline - self.acquisition.clock()).total_seconds())))
                if len(results) != len(eligible) or any(s not in {'pending', 'interpret', 'informational', 'irrelevant'} for s in results):
                    raise ValueError('project_triage_result_invalid')
                outcomes.update(zip((r['evidence_id'] for r in eligible), results))
                return CheckStep()
            try:
                ctx.perform('analysis', call)
                telemetry['status'] = 'answered'
            except CheckStopped:
                # Preserve candidates even when the durable call budget is exhausted.
                ctx.guard()
                telemetry = dict(version=VERSION, contract_digest=CONTRACT_DIGEST, questions=0, usage={}, status='budget')
            except Exception:
                ctx.guard()
                telemetry = dict(version=VERSION, contract_digest=CONTRACT_DIGEST, questions=4 * len(eligible), usage={}, status='unavailable')
                outcomes = {r['evidence_id']: 'pending' for r in batch}
        elif eligible:
            telemetry['status'] = 'required_system_one_unavailable'
        ctx.guard()
        from alphonse.agent_v2.services.source_acquisition import digest
        telemetry['batch_id'] = digest([run_id, [r['evidence_id'] for r in batch], VERSION])
        # Scope and retention may have changed during the decision call.
        current = {r['evidence_id']: r for r in self.acquisition.evidence(session, run_id)}
        if any(r['evidence_id'] not in current or current[r['evidence_id']]['content_digest'] != r['content_digest'] for r in batch):
            raise CheckStopped('blocked', 'project_triage_evidence_changed')
        with self.acquisition.dispatcher.store._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            fence = conn.execute('''SELECT 1 FROM v2_project_check_runs WHERE check_run_id=?
                AND state='leased' AND generation=? AND lease_owner=? AND lease_until>?''',
                (run_id, ctx.run['generation'], ctx.run['lease_owner'], self.acquisition.clock().isoformat())).fetchone()
            if not fence:
                raise CheckStopped('blocked', 'lease_lost')
            for record in batch:
                receipt = dict(evidence_id=record['evidence_id'], content_digest=record['content_digest'],
                    source_version=record['version'], status=outcomes[record['evidence_id']],
                    telemetry=telemetry)
                conn.execute('INSERT OR IGNORE INTO v2_project_triage VALUES(?,?,?)',
                    (run_id, record['evidence_id'], json.dumps(receipt)))
        resolved = [r['evidence_id'] for r in batch if outcomes[r['evidence_id']] != 'pending']
        self.acquisition.checkpoint_processing(session, run_id, stage='classification', evidence_ids=resolved)
        ctx.run['coverage']['processing'] = 'pending' if any(v == 'pending' for v in outcomes.values()) or len(remaining) > len(batch) else 'classified'
        return CheckStep(analysis_checkpoint={'triage_version': VERSION})
