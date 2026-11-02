"""Audience-scoped commitments derived by System Two from bounded evidence.

This is a trusted service, not an extractor or a permission source. Callers pass
structured interpretations, never executable instructions. Only opaque finding
references should enter general memory, task checkpoints, caches or UI indexes;
resolve them here at use time. No mailbox content is copied to the memory ledger.
"""
from datetime import datetime, timedelta
import json
from zoneinfo import ZoneInfo

from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.services.source_acquisition import digest
from alphonse.agent_v2.services.project_check_worker import instant


class FindingService:
    def __init__(self, acquisition, outbox, resolver):
        self.acquisition, self.outbox, self.resolver = acquisition, outbox, resolver
        self.store = acquisition.dispatcher.store
        with self.store._connect() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS v2_findings (
                    finding_id TEXT PRIMARY KEY, identity_key TEXT NOT NULL UNIQUE,
                    definition TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS v2_finding_versions (
                    finding_id TEXT NOT NULL, version INTEGER NOT NULL,
                    definition TEXT NOT NULL, PRIMARY KEY(finding_id,version));
                CREATE TABLE IF NOT EXISTS v2_finding_observations (
                    observation_id TEXT PRIMARY KEY, finding_id TEXT NOT NULL);
            ''')
        outbox.finding_delivery_guard = self.delivery_allowed

    def _load(self, finding_id):
        with self.store._connect() as conn:
            row = conn.execute('SELECT definition FROM v2_findings WHERE finding_id=?', (finding_id,)).fetchone()
        if row is None:
            raise ValueError('finding_missing')
        return json.loads(row[0])

    def _authorize(self, finding, recipient=None):
        if finding['state'] == 'redacted':
            raise PermissionError('finding_unavailable')
        policies = self.acquisition.dispatcher.policies
        actor = AccountSession(finding['executing_user_id'])
        policy = policies._managed(actor, finding['policy_id'])
        # Retained findings outlive raw cache bytes and an inactive project.
        # Grant revocation/narrowing still fences every historical read.
        policies._sources(actor, dict(policy, source_bindings=finding['source_bindings'],
                          recipient_user_ids=finding['audience_user_ids']), pinned=True)
        if instant(finding['retention_deadline']) <= self.acquisition.clock():
            raise PermissionError('finding_retention_expired')
        if not set(finding['audience_user_ids']) <= set(policy['recipient_user_ids']):
            raise PermissionError('finding_audience_denied')
        if recipient is not None and recipient not in finding['audience_user_ids']:
            raise PermissionError('finding_audience_denied')
        # Guard resolves every contributing source grant, including recipients.
        with self.store._connect() as conn:
            for ref in finding['source_refs']:
                row = conn.execute('SELECT definition FROM v2_source_evidence WHERE evidence_id=?',
                                   (ref['evidence_id'],)).fetchone()
                evidence = json.loads(row[0]) if row else {}
                if not evidence or evidence.get('state') == 'deleted' or evidence.get('deleted'):
                    raise PermissionError('finding_source_unavailable')
                run = self.acquisition.dispatcher.get(evidence['check_run_id'])
                if run['outcome'] == 'cancelled':
                    raise PermissionError('finding_source_cancelled')
        return policy

    @staticmethod
    def _interpretation(value):
        allowed = {'kind', 'subject_id', 'commitment_id', 'summary', 'facts', 'uncertainty',
                   'due_at', 'timezone', 'state', 'quoted', 'identity_uncertain'}
        if not isinstance(value, dict) or set(value) - allowed:
            raise ValueError('finding_interpretation_invalid')
        result = dict(kind=value.get('kind'), subject_id=value.get('subject_id', ''),
            commitment_id=value.get('commitment_id', ''), summary=value.get('summary', ''),
            facts=value.get('facts', []), uncertainty=value.get('uncertainty', []),
            due_at=value.get('due_at'), timezone=value.get('timezone'), state=value.get('state', 'open'),
            quoted=value.get('quoted', False), identity_uncertain=value.get('identity_uncertain', False))
        if (result['kind'] not in {'assignment', 'invoice', 'commitment'}
                or result['state'] not in {'open', 'unresolved', 'withdrawn'}
                or type(result['quoted']) is not bool or type(result['identity_uncertain']) is not bool
                or not isinstance(result['summary'], str) or not 0 < len(result['summary']) <= 1000
                or any(not isinstance(result[k], str) or len(result[k]) > 200 for k in ('subject_id', 'commitment_id'))
                or not isinstance(result['facts'], list) or len(result['facts']) > 20
                or not isinstance(result['uncertainty'], list) or len(result['uncertainty']) > 20
                or any(not isinstance(x, str) or len(x) > 500 for x in result['uncertainty'])):
            raise ValueError('finding_interpretation_invalid')
        if result['timezone'] is not None:
            ZoneInfo(result['timezone'])
        if result['due_at'] is not None:
            # Date-only deadlines stay dates. Unknown timezone stays unknown.
            datetime.fromisoformat(result['due_at'].replace('Z', '+00:00'))
        for fact in result['facts']:
            if (not isinstance(fact, dict) or set(fact) != {'predicate', 'value', 'basis', 'unit'}
                    or fact['predicate'] not in {'deadline', 'overdue_notice', 'payment_report',
                        'assignment_presence', 'calendar_presence', 'submission_report', 'grade', 'requirement'}
                    or fact['basis'] not in {'sender_reported', 'source_observed'}
                    or not isinstance(fact['value'], (str, int, float, bool))
                    or fact['unit'] not in {None, 'points', 'percent', 'currency'}):
                raise ValueError('finding_fact_invalid')
            if fact['predicate'] not in {'assignment_presence', 'calendar_presence', 'grade'} and fact['basis'] != 'sender_reported':
                raise ValueError('finding_report_is_not_verified_outcome')
            if fact['predicate'] == 'grade' and fact['unit'] not in {'points', 'percent'}:
                raise ValueError('finding_grade_unit_required')
        result['facts'] = sorted(result['facts'], key=lambda fact: json.dumps(fact, sort_keys=True))
        result['uncertainty'] = sorted(set(result['uncertainty']))
        if len(json.dumps(result, allow_nan=False).encode()) > 12000:
            raise ValueError('finding_interpretation_too_large')
        if not result['subject_id'] or not result['commitment_id'] or result['identity_uncertain'] or result['quoted']:
            result['state'] = 'unresolved'
        return result

    def record(self, session, run_id, *, evidence_ids, interpretation, audience_user_ids):
        """Persist one bounded interpretation; replaying it is a no-op.

        Identity keys must be source-backed canonical child/person AND assignment,
        invoice or commitment IDs, never a title-only semantic similarity match.
        Uncertain identity is isolated per evidence/interpretation for later review.
        """
        value = self._interpretation(interpretation)
        if not evidence_ids or len(evidence_ids) > 20 or not audience_user_ids:
            raise ValueError('finding_sources_and_audience_required')
        records = {r['evidence_id']: r for r in self.acquisition.evidence(session, run_id)}
        if not set(evidence_ids) <= records.keys():
            raise PermissionError('finding_evidence_denied')
        run = self.acquisition.dispatcher.get(run_id)
        policy = self.acquisition._policy(run)
        audience = sorted(set(audience_user_ids))
        if not set(audience) <= set(policy['recipient_user_ids']):
            raise PermissionError('finding_audience_denied')
        refs = []
        for eid in sorted(set(evidence_ids)):
            r = records[eid]
            if not set(audience) <= set(r['audience_user_ids']):
                raise PermissionError('finding_audience_denied')
            refs.append({k: r[k] for k in ('evidence_id', 'connection_id', 'resource', 'source_id',
                'version', 'source_timestamp', 'observed_at', 'safe_url', 'check_run_id')})
        observation_id = digest([run_id, refs, value, audience])
        identity = [policy['policy_id'], policy['project_id'], audience, value['kind'],
                    value['subject_id'], value['commitment_id']]
        if value['state'] == 'unresolved':
            identity.append(digest([refs, value]))
        identity_key = digest(identity)
        fid = 'finding_' + identity_key
        timestamp = max(instant(r['source_timestamp']) for r in refs).isoformat()
        with self.store._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            old_row = conn.execute('SELECT definition FROM v2_findings WHERE identity_key=?', (identity_key,)).fetchone()
            old = json.loads(old_row[0]) if old_row else None
            if old and (old['state'] == 'redacted' or instant(old['retention_deadline']) <= self.acquisition.clock()):
                raise PermissionError('finding_unavailable')
            # Version only material facts, not prose or coverage/transport metadata.
            material = {k: value[k] for k in ('facts', 'due_at', 'timezone', 'state', 'uncertainty')}
            changed = old is None or digest(material) != old['material_digest']
            newer = old is None or instant(timestamp) > instant(old['source_timestamp'])
            accept = old is None or (changed and newer and not value['quoted'])
            if old is None:
                finding = dict(finding_id=fid, version=0, policy_id=policy['policy_id'],
                    policy_revision=policy['revision'], project_id=policy['project_id'],
                    executing_user_id=session.actor_user_id, audience_user_ids=audience,
                    goal_digest=policy['goal_digest'], source_bindings=policy['source_bindings'],
                    retention_deadline=(instant(policy['duration']['ends_at']) + timedelta(days=90)
                        if policy['duration']['ends_at'] else self.acquisition.clock() + timedelta(days=365)).isoformat(),
                    source_refs=[], correction_of=None, related_finding_ids=[])
            else:
                finding = old
            existing = {r['evidence_id'] for r in finding['source_refs']}
            finding['source_refs'].extend(r for r in refs if r['evidence_id'] not in existing)
            if accept:
                previous = finding['version']
                finding.update(value, version=previous + 1, material_digest=digest(material),
                    source_timestamp=timestamp, coverage=run['coverage'], supporting_source_refs=refs,
                    correction_of={'finding_id': fid, 'version': previous} if previous else None)
                conn.execute('INSERT INTO v2_finding_versions VALUES(?,?,?)',
                             (fid, finding['version'], json.dumps(finding)))
            conn.execute('INSERT OR REPLACE INTO v2_findings VALUES(?,?,?)', (fid, identity_key, json.dumps(finding)))
            conn.execute('INSERT OR IGNORE INTO v2_finding_observations VALUES(?,?)', (observation_id, fid))
        # Independent CAPD checkpoint; failure here replays the same observation.
        self.acquisition.checkpoint_processing(session, run_id, stage='capd', evidence_ids=evidence_ids)
        return self.reference(finding)

    @staticmethod
    def reference(finding):
        return {'finding_id': finding['finding_id'], 'version': finding['version']}

    def view(self, session, finding_id, *, version=None):
        finding = self._load(finding_id)
        self._authorize(finding, session.actor_user_id)
        if version is not None:
            with self.store._connect() as conn:
                row = conn.execute('SELECT definition FROM v2_finding_versions WHERE finding_id=? AND version=?',
                                   (finding_id, version)).fetchone()
            if row is None:
                raise ValueError('finding_version_missing')
            return json.loads(row[0])
        return finding

    def notify(self, session, finding_id):
        finding = self._load(finding_id)
        if session.actor_user_id != finding['executing_user_id']:
            raise PermissionError('finding_actor_denied')
        self._authorize(finding)
        self.acquisition.dispatcher.policies.guard(finding['policy_id'],
            expected_revision=finding['policy_revision'], now=self.acquisition.clock())
        if finding['quoted'] or finding['identity_uncertain'] or not finding['subject_id'] or not finding['commitment_id']:
            return {recipient: {'state': 'pending_identity_review'} for recipient in finding['audience_user_ids']}
        result = {}
        for recipient in finding['audience_user_ids']:
            self._authorize(finding, recipient)
            address = self.resolver.resolve_outbound_address(alphonse_user_id=recipient)
            if not address.resolved or address.address is None:
                result[recipient] = {'state': 'blocked', 'reason': 'preferred_delivery_not_found'}
                continue
            key = f"finding:{finding_id}:{finding['version']}:{recipient}"
            source = max(finding['supporting_source_refs'], key=lambda r: instant(r['source_timestamp']))
            text = self._message(finding, source)
            message = self.outbox.enqueue(address=address.address, message=text, kind='finding',
                audience_user_id=recipient, project_id=finding['project_id'], idempotency_key=key,
                metadata={'finding_id': finding_id, 'finding_version': finding['version']})
            result[recipient] = {'state': message.metadata.get('finding_delivery_state', message.status),
                                 'outbox_message_id': message.outbox_message_id}
        return result

    def delivery_status(self, session, finding_id):
        finding = self._load(finding_id)
        operator = session.actor_user_id == finding['executing_user_id']
        self._authorize(finding, None if operator else session.actor_user_id)
        with self.outbox._connect() as conn:
            rows = conn.execute("""SELECT audience_user_id,status,metadata_json,outbox_message_id,
                provider_message_id FROM v2_outbox WHERE kind='finding'
                AND json_extract(metadata_json, '$.finding_id')=?
                AND json_extract(metadata_json, '$.finding_version')=?""",
                (finding_id, finding['version'])).fetchall()
        return {row['audience_user_id']: dict(
            state=json.loads(row['metadata_json']).get('finding_delivery_state', row['status']),
            outbox_message_id=row['outbox_message_id'], provider_message_id=row['provider_message_id'])
            for row in rows if operator or row['audience_user_id'] == session.actor_user_id}

    @staticmethod
    def _message(finding, source):
        parts = [f"{'Correction' if finding['correction_of'] else 'Finding'}: {finding['summary']}",
                 'Source-backed report; presence is not completion or payment verification.']
        parts.extend(f"{fact['predicate']}: {fact['value']} ({fact['unit'] or 'no unit'}; {fact['basis']})" for fact in finding['facts'])
        if finding['due_at']:
            parts.append(f"Due: {finding['due_at']} ({finding['timezone'] or 'timezone unknown'})")
        parts.append(f"Source: {source['safe_url'] or source['source_id']} ({source['source_timestamp']})")
        parts.extend('Uncertainty: ' + item for item in finding['uncertainty'])
        coverage = finding['coverage']
        if coverage.get('gaps') or not coverage.get('acquisition_complete'):
            parts.append('Coverage is partial; unseen sources and missing outcomes remain unknown.')
        return '\n'.join(parts)

    def delivery_allowed(self, message):
        finding = self._load(message.metadata['finding_id'])
        self._authorize(finding, message.audience_user_id)
        self.acquisition.dispatcher.policies.guard(finding['policy_id'],
            expected_revision=finding['policy_revision'], now=self.acquisition.clock())
        return finding['version'] == message.metadata['finding_version']

    def purge(self):
        """Invalidate every derived version and pending/delivered local projection.

        General memory/checkpoints receive IDs only, so invalidation at resolution
        also covers cached references without duplicating confidential payloads.
        """
        with self.store._connect() as conn:
            rows = conn.execute('SELECT definition FROM v2_findings').fetchall()
        for row in rows:
            finding = json.loads(row[0])
            try:
                self._authorize(finding)
                continue
            except (PermissionError, ValueError, KeyError, OSError):
                pass
            tombstone = {k: finding[k] for k in ('finding_id', 'version', 'policy_id', 'policy_revision',
                         'project_id', 'executing_user_id', 'audience_user_ids')}
            tombstone.update(state='redacted', source_refs=[])
            with self.store._connect() as conn:
                conn.execute('UPDATE v2_findings SET definition=? WHERE finding_id=?',
                             (json.dumps(tombstone), finding['finding_id']))
                conn.execute('DELETE FROM v2_finding_versions WHERE finding_id=?', (finding['finding_id'],))
            self.outbox.redact_finding(finding['finding_id'])
