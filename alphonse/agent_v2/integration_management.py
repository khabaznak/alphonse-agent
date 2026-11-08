"""Authenticated, secret-free management shared by Desktop and conversation tools."""
from datetime import datetime, timezone
import json
import secrets
import threading
import time
from uuid import uuid4

from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.project_checks import _digest


class IntegrationManagement:
    def __init__(self, runtime, *, oauth=None, dispatcher=None, findings=None):
        self.runtime = runtime
        self.accounts = runtime.project_checks.bindings.accounts
        self.policies = runtime.project_checks
        self.oauth = oauth
        self.oauth_factory = None
        self.dispatcher = dispatcher
        self.findings = findings
        self._reviews = {}
        self._lock = threading.RLock()

    def _session(self, session):
        if not isinstance(session, AccountSession):
            raise PermissionError('management_authenticated_session_required')
        self.accounts._active(session.actor_user_id)
        if self.oauth is None and self.oauth_factory:
            self.oauth = self.oauth_factory(session)

    def catalog(self, session):
        self._session(session)
        actor = session.actor_user_id
        owned = []
        for record in self.accounts.integrations.list():
            if record.connection.get('owner_user_id') == actor:
                health = self.oauth.health(session, record.integration_id) if self.oauth else dict(
                    connection_id=record.integration_id, display_name=record.display_name,
                    provider=record.provider_key, **{k: record.connection[k] for k in
                        ('revision', 'status', 'health_code', 'capabilities')})
                health.pop('granted_scopes', None)
                owned.append(health)
        with self.accounts.integrations._connect() as conn:
            grants = [json.loads(r[0]) for r in conn.execute('SELECT definition FROM v2_account_grants')]
        grants = [g for g in grants if actor in (g['grantor_user_id'], g['beneficiary_user_id'])]
        # A household directory is needed for explicit recipients; no addresses or profiles.
        members = [dict(user_id=u.user_id, display_name=u.display_name) for u in
                   self.runtime.user_store.list_users() if u.is_active]
        from alphonse.agent_v2.integrations.oauth import SCOPES
        clients = self.oauth.clients if self.oauth else {}
        status = getattr(self.oauth, 'provider_status', {})
        sources = getattr(self.oauth, 'provider_sources', {})
        providers = [dict(provider=p, capabilities=list(SCOPES[p]),
                          status=status.get(p, 'available' if p in clients else 'unconfigured'),
                          source=sources.get(p, 'distribution'))
                     for p in SCOPES]
        return dict(actor_user_id=actor, can_configure_oauth=self.runtime.user_store.is_admin(actor), connections=owned, grants=grants, members=members,
                    providers=providers, sources=self.sources(session), skills=[dict(skill_id=s.skill_id, name=s.name)
                        for s in self.runtime.skill_store.list_skills() if self.runtime.skill_store.can_use(s.skill_id, actor)],
                    bindings=self.binding_options(session), projects=[dict(project_id=p.project_id, name=p.name)
                        for p in self.policies.projects.list_visible_projects(actor) if p.status == 'active'])

    def binding_options(self, session):
        result = []
        with self.accounts.integrations._connect() as conn:
            rows = conn.execute("SELECT * FROM v2_skill_bindings WHERE kind='account' AND enabled=1").fetchall()
        for row in rows:
            requirements = json.loads(row['requirements'])
            if row['owner_id'] != session.actor_user_id or not requirements.get('grant_id'):
                continue
            if not self.runtime.skill_store.can_use(row['skill_id'], session.actor_user_id):
                continue
            result.append(dict(binding_id=row['binding_id'], skill_id=row['skill_id'],
                               grant_id=requirements['grant_id']))
        return result

    def sources(self, session):
        """Only explicit actor choices backed by a usable skill and their grant."""
        self._session(session)
        bindings = self.policies.bindings
        result = []
        with bindings.integrations._connect() as conn:
            rows = conn.execute('''SELECT c.skill_id,c.name,b.* FROM v2_skill_binding_choices c
                JOIN v2_skill_bindings b ON b.binding_id=c.binding_id
                WHERE c.actor_id=? AND b.kind='account' AND b.enabled=1''', (session.actor_user_id,)).fetchall()
        for row in rows:
            if not bindings.skills.can_use(row['skill_id'], session.actor_user_id):
                continue
            requirements = json.loads(row['requirements'])
            grant_id = requirements.get('grant_id')
            if not grant_id:
                continue
            with bindings.integrations._connect() as conn:
                grant_row = conn.execute('SELECT definition FROM v2_account_grants WHERE grant_id=?', (grant_id,)).fetchone()
            grant = json.loads(grant_row[0]) if grant_row else {}
            if grant.get('beneficiary_user_id') != session.actor_user_id or grant.get('revoked_at'):
                continue
            for operation in grant['operations']:
                try:
                    bindings.resolve_account(session, skill_id=row['skill_id'], name=row['name'],
                        operation=operation, resources=grant['resources'], recipient_user_ids=[],
                        project_id=grant['project_id'])
                except (ValueError, PermissionError, KeyError):
                    continue
                record = bindings.integrations.get(row['target'])
                result.append(dict(skill_id=row['skill_id'], role=row['name'], binding_id=row['binding_id'],
                    connection_id=row['target'], grant_id=grant_id, operation=operation,
                    resources=grant['resources'], recipient_user_ids=grant['recipient_user_ids'],
                    project_id=grant['project_id'], display_name=record.display_name))
        return result

    def project(self, session, project_id):
        self._session(session)
        project = self.policies.projects.get_project(project_id, requester_user_id=session.actor_user_id, include_archived=True)
        if not project:
            raise PermissionError('project_check_project_unavailable')
        with self.policies.projects._connect() as conn:
            ids = [r[0] for r in conn.execute('SELECT policy_id FROM v2_project_check_policies WHERE project_id=? AND executing_user_id=?',
                (project_id, session.actor_user_id))]
        policies = [self.policies.get(session, pid) for pid in ids]
        result = []
        now = datetime.now(timezone.utc)
        for policy in policies:
            runs = self.dispatcher.status(session, policy['policy_id'], now=now) if self.dispatcher else []
            # Never serialize checkpoints, raw evidence, provider cursors or exception text.
            runs = [{k: r.get(k) for k in ('check_run_id', 'state', 'outcome', 'error_code',
                'window_start', 'window_end', 'coverage', 'stale', 'offline', 'items', 'acquisition_checkpoint',
                'analysis_checkpoint', 'processing_details_expired')} for r in runs]
            for run in runs:
                for key in ('acquisition_checkpoint', 'analysis_checkpoint'):
                    checkpoint = run.pop(key, None)
                    run[key.replace('_checkpoint', '_state')] = 'checkpoint_saved' if checkpoint else 'not_started'
                run['pending_analysis'] = None if run.get('processing_details_expired') else False
                if not run.get('processing_details_expired') and self.findings and run.get('items') and not run.get('coverage', {}).get('triage_no_match'):
                    with self.dispatcher.store._connect() as conn:
                        row = conn.execute("SELECT definition FROM v2_source_processing WHERE check_run_id=? AND stage='capd'",
                                           (run['check_run_id'],)).fetchone()
                        evidence_ids = {r[0] for r in conn.execute('SELECT evidence_id FROM v2_source_evidence WHERE check_run_id=?', (run['check_run_id'],))}
                        ignored = set()
                        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_project_triage'").fetchone():
                            ignored = {r[0] for r in conn.execute("SELECT evidence_id FROM v2_project_triage WHERE check_run_id=? AND json_extract(definition,'$.status')='irrelevant'", (run['check_run_id'],))}
                    completed = set(json.loads(row[0])) if row else set()
                    run['pending_analysis'] = bool(evidence_ids - completed - ignored)
            next_run = None
            if self.dispatcher:
                with self.dispatcher.store._connect() as conn:
                    row = conn.execute('''SELECT next_run_at FROM v2_scheduled_tasks WHERE job_kind='project_check'
                        AND status='active' AND json_extract(payload_json,'$.policy_id')=?
                        AND json_extract(payload_json,'$.policy_revision')=?''',
                        (policy['policy_id'], policy['revision'])).fetchone()
                next_run = row[0] if row else None
            result.append(dict(policy=policy, runs=runs, next_run_at=next_run))
        return dict(checks=result, findings=self.finding_views(session, project_id), sources=[s for s in self.sources(session)
                    if s['project_id'] in (None, project_id)])

    def finding_views(self, session, project_id=None):
        self._session(session)
        findings = []
        if self.findings:
            with self.findings.store._connect() as conn:
                if project_id:
                    ids = [r[0] for r in conn.execute("SELECT finding_id FROM v2_findings WHERE json_extract(definition,'$.project_id')=?", (project_id,))]
                else:
                    ids = [r[0] for r in conn.execute('SELECT finding_id FROM v2_findings')]
            for fid in ids:
                try:
                    value = self.findings.view(session, fid)
                    deliveries = self.findings.delivery_status(session, fid)
                except (PermissionError, ValueError, KeyError):
                    continue
                findings.append(dict(finding_id=fid, summary=value['summary'], state=value['state'],
                    due_at=value['due_at'], timezone=value['timezone'], uncertainty=value['uncertainty'],
                    coverage=value['coverage'], deliveries=deliveries))
        return findings

    def review(self, session, *, definition, policy_id=None, expected_revision=0, channel='desktop'):
        self._session(session)
        value = self.policies._validate(definition)
        if not value['query']['text'].strip() or not value['recipient_user_ids']:
            raise ValueError('management_goal_and_recipients_required')
        snapshot = self.policies._snapshot(session, value['project_id'])
        if policy_id:
            old = self.policies._managed(session, policy_id)
            if old['revision'] != expected_revision or old['project_id'] != value['project_id']:
                raise ValueError('project_check_revision_conflict')
        elif expected_revision != 0:
            raise ValueError('project_check_revision_conflict')
        pinned = self.policies._sources(session, value, pinned=False)
        self.policies._skills(session, value)
        self.policies._duration(dict(value, goal_snapshot=snapshot), datetime.now(timezone.utc), require_started=False)
        summary = dict(goal=snapshot, **value)
        receipt = dict(definition=value, policy_id=policy_id, expected_revision=expected_revision,
                       snapshot_digest=_digest(snapshot), sources_digest=_digest(pinned),
                       project_revision=self.policies._project_revision(value['project_id']))
        with self._lock:
            self._reviews = {k: v for k, v in self._reviews.items() if v['expires'] > time.monotonic()}
            if len(self._reviews) >= 128:
                raise ValueError('management_reviews_busy')
            token = secrets.token_urlsafe(32)
            self._reviews[token] = dict(receipt, actor=session.actor_user_id, channel=channel, expires=time.monotonic() + 300)
        return dict(review_id=token, summary=summary, expires_in=300)

    def apply_review(self, session, review_id, *, activate, channel='desktop'):
        self._session(session)
        if type(activate) is not bool:
            raise ValueError('management_activation_boolean_required')
        with self._lock:
            review = self._reviews.get(review_id)
            if (not review or review['actor'] != session.actor_user_id or review['channel'] != channel
                    or review['expires'] <= time.monotonic()):
                raise PermissionError('management_review_expired_or_denied')
            value = review['definition']
            if (review['snapshot_digest'] != _digest(self.policies._snapshot(session, value['project_id']))
                    or review['project_revision'] != self.policies._project_revision(value['project_id'])
                    or review['sources_digest'] != _digest(self.policies._sources(session, value, pinned=False))):
                raise ValueError('management_review_stale')
            result = self.policies.save(session, definition=value, policy_id=review['policy_id'],
                expected_revision=review['expected_revision'], activate=activate,
                authorization_reference=f'{channel}:review:{uuid4().hex}' if activate else '', reviewed_authority=review)
            del self._reviews[review_id]
        if self.dispatcher:
            self.dispatcher.sync(now=datetime.now(timezone.utc))
        return result

    def request(self, session, action, values, *, channel='desktop'):
        self._session(session)
        if action == 'findings':
            return {'findings': self.finding_views(session)}
        if action == 'catalog':
            return self.catalog(session)
        if action == 'project':
            return self.project(session, **values)
        if action == 'review':
            return self.review(session, **values, channel=channel)
        if action == 'apply_review':
            return self.apply_review(session, **values, channel=channel)
        if action == 'set_status':
            result = self.policies.set_status(session, **values)
            if self.dispatcher:
                self.dispatcher.sync(now=datetime.now(timezone.utc))
            return result
        if action == 'save_grant':
            return self.accounts.save_grant(session, **values)
        if action == 'revoke_grant':
            self.accounts.revoke_grant(session, **values)
            return {'revoked': True}
        if action == 'bind_grant':
            # Owner creates a binding; beneficiary separately selects it. No implicit choice.
            grant = next((g for g in self.catalog(session)['grants'] if g['grant_id'] == values['grant_id']
                          and g['grantor_user_id'] == session.actor_user_id), None)
            if not grant:
                raise PermissionError('account_owner_required')
            binding_id = 'account_' + uuid4().hex
            self.policies.bindings.configure(actor_id=session.actor_user_id, binding_id=binding_id,
                skill_id=values['skill_id'], owner_id=grant['beneficiary_user_id'], kind='account',
                target=grant['connection_id'], grant_id=grant['grant_id'], capabilities=tuple(grant['operations']))
            return {'binding_id': binding_id}
        if action == 'select_binding':
            self.policies.bindings.select(actor_id=session.actor_user_id, **values)
            return {'selected': True}
        if action in {'begin', 'complete', 'pause', 'disconnect', 'retry_deletion'}:
            if not self.oauth:
                raise ValueError('oauth_setup_unavailable')
            return getattr(self.oauth, action)(session, **values)
        raise ValueError('management_action_invalid')
